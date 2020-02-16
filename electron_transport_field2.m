function [ gridE_r,gridE_z,gridH_theta,gridE_r_PML0,gridE_z_PML0,gridD_r_PML0,gridD_z_PML0,gridH_theta_PML0,gridE_r_PMLn,gridE_z_PMLn,gridD_r_PMLn,gridD_z_PMLn,gridH_theta_PMLn,x_e,y_e,z_e,vx_e,vy_e,vz_e,x_ion1,y_ion1,z_ion1,vx_ion1,vy_ion1,vz_ion1,x_ion2,y_ion2,z_ion2,vx_ion2,vy_ion2,vz_ion2,PHI,RHO] = electron_transport_field2( x,y,z,vx,vy,vz,v,charge,INDEX,VC,FN,x_e,y_e,z_e,r_e,phi_e,vx_e,vy_e,vz_e,v_e,x_ion1,y_ion1,z_ion1,vx_ion1,vy_ion1,vz_ion1,x_ion2,y_ion2,z_ion2,vx_ion2,vy_ion2,vz_ion2,n_e,dt_e,dt,m,m_e,r_divide,phi_divide,z_divide,dphi,Rp,Lp,cellnumxy,sum_cell,phimax,gridE_r,gridE_z,gridH_theta,gridE_r_PML0,gridE_z_PML0,gridD_r_PML0,gridD_z_PML0,gridH_theta_PML0,gridE_r_PMLn,gridE_z_PMLn,gridD_r_PMLn,gridD_z_PMLn,gridH_theta_PMLn,num_mol_density,delta,PHI,RHO )
c=3E8;
T=m_e*c^2*(1./sqrt(1-v_e.^2/c^2)-1)/1.6E-19/1000; %in keV;
%occupy=ones(n_e,1);   %denote whether the electron exists
%% Read table for elastic scattering 
% Find total cross section: 
% For energy > 50 eV
T_el = readtable("TotalElasticCS_Nitrogen.csv");
t_el = table2array(T_el);
% For 1 eV < energy < 50 eV
T_sel = readtable("TotalElasticCS_smalleV_Nitrogen.csv");
t_sel = table2array(T_sel);
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
k_B = 1.38064853E-23;  
m_ion = 4.65E-26;
e = 1.6*10^(-19);
temperature = 300;
time=0;

delta_t = 1*10^(-13);
delta_r = delta;
delta_z = delta;


H_theta_gridr = delta_r/2:delta_r:Rp-delta_r/2;
H_theta_gridz = delta_z/2:delta_z:Lp-delta_z/2;

H_theta_lenr = length(H_theta_gridr);
H_theta_lenz = length(H_theta_gridz);
H_theta_len = length(H_theta_gridr)*length(H_theta_gridz);

[H_theta_R,H_theta_Z] = meshgrid(H_theta_gridr,H_theta_gridz);
H_theta_R = H_theta_R';
H_theta_Z = H_theta_Z';



E_r_gridr = delta_r/2:delta_r:Rp-delta_r/2;
E_r_gridz = 0:delta_z:Lp;

E_r_lenr = length(E_r_gridr);
E_r_lenz = length(E_r_gridz);
E_r_len = length(E_r_gridr)*length(E_r_gridz);

[E_r_R,E_r_Z] = meshgrid(E_r_gridr,E_r_gridz);
E_r_R = E_r_R';
E_r_Z = E_r_Z';



E_z_gridr = 0:delta_r:Rp;
E_z_gridz = delta_z/2:delta_z:Lp-delta_z/2;

E_z_lenr = length(E_z_gridr);
E_z_lenz = length(E_z_gridz);
E_z_len = length(E_z_gridr)*length(E_z_gridz);

[E_z_R,E_z_Z] = meshgrid(E_z_gridr,E_z_gridz);
E_z_R = E_z_R';
E_z_Z = E_z_Z';
gridE_r = zeros(E_r_lenr,E_r_lenz);
gridj_r = zeros(E_r_lenr,E_r_lenz);
gridE_z = zeros(E_z_lenr,E_z_lenz);
gridj_z = zeros(E_z_lenr,E_z_lenz);
gridH_theta = zeros(H_theta_lenr,H_theta_lenz,1);



gridr = 0:delta:Rp;
gridz = 0:delta_z:Lp;
len=length(gridr)*length(gridz);
[R,Z] = meshgrid(gridr,gridz);
R = R';
Z = Z';
cycle_time = round(dt_e/delta_t)
n_e = length(x_e);
threshold = 15.6*10^(-3); % Ionization threshold, unit : eV
plotE = [0 0];
large_cycle_time = 100;
kk = 0;
tt = 0;
iii = 0;
message_E_r_matrix = [];
message_E_z_matrix = [];
message_H_matrix = [];
for time = 1:large_cycle_time
    
    tic
    fprintf('t=%d\n',time);
    
    %% motion in electricmagnetic field
    if n_e>0
        
        electron_position = zeros(length(x_e),3);
        electron_position(:,1) = x_e;
        electron_position(:,2) = y_e;
        electron_position(:,3) = z_e;
        electron_velocity = zeros(length(x_e),3);
        electron_velocity(:,1) = vx_e;
        electron_velocity(:,2) = vy_e;
        electron_velocity(:,3) = vz_e;
        ion1_position = zeros(length(x_ion1),3);
        ion1_position(:,1) = x_ion1;
        ion1_position(:,2) = y_ion1;
        ion1_position(:,3) = z_ion1;
        ion1_velocity = zeros(length(x_ion1),3);
        ion1_velocity(:,1) = vx_ion1;
        ion1_velocity(:,2) = vy_ion1;
        ion1_velocity(:,3) = vz_ion1;
        %ion1_position = gpuArray(ion1_position);
        %ion1_velocity = gpuArray(ion1_velocity);
        
        ion2_position = zeros(length(x_ion2),3);
        ion2_position(:,1) = x_ion2;
        ion2_position(:,2) = y_ion2;
        ion2_position(:,3) = z_ion2;
        ion2_velocity = zeros(length(x_ion2),3);
        ion2_velocity(:,1) = vx_ion2;
        ion2_velocity(:,2) = vy_ion2;
        ion2_velocity(:,3) = vz_ion2;
        %ion2_position = gpuArray(ion2_position);
        %ion2_velocity = gpuArray(ion2_velocity);
        fprintf('EM field:\n');
        duandian = 0;
        tic
        %electron_position = gpuArray(electron_position);
        %electron_velocity = gpuArray(electron_velocity);
        %gridE = gpuArray(gridE);
        %gridB_theta = gpuArray(gridB_theta);
        for timee=1:cycle_time
            duandian = duandian+1;
            
            [electron_position,electron_velocity,ion1_position,ion1_velocity,ion2_position,ion2_velocity,gridE_r,gridE_z,gridH_theta,gridE_r_PML0,gridE_z_PML0,gridD_r_PML0,gridD_z_PML0,gridH_theta_PML0,gridE_r_PMLn,gridE_z_PMLn,gridD_r_PMLn,gridD_z_PMLn,gridH_theta_PMLn,energy_total,energy_of_electric_field,energy_of_magnetic_field,energy_of_particle] = electricmagneticfield3(electron_position,electron_velocity,ion1_position,ion1_velocity,ion2_position,ion2_velocity,gridE_r,gridE_z,gridH_theta,gridE_r_PML0,gridE_z_PML0,gridD_r_PML0,gridD_z_PML0,gridH_theta_PML0,gridE_r_PMLn,gridE_z_PMLn,gridD_r_PMLn,gridD_z_PMLn,gridH_theta_PMLn,delta,delta_t,Rp,Lp,phimax,FN,duandian);
            
            %histogram(sqrt(electron_velocity(:,1).^2+electron_velocity(:,2).^2+electron_velocity(:,3).^2));
            %histogram(1/2*m_e/e*(electron_velocity(:,1).^2+electron_velocity(:,2).^2+electron_velocity(:,3).^2));
            %kk = kk+1;
            %histogram(1/2*m_ion/e*(ion2_velocity(:,1).^2+ion2_velocity(:,2).^2+ion2_velocity(:,3).^2));
            %fmat(kk,:) = getframe;
            %{
            if mod(timee,2) == 1
                iii = iii+1;
                message_E_r_matrix(iii,:,:) = gridE_r;
                message_E_z_matrix(iii,:,:) = gridE_z;
                message_H_matrix(iii,:,:) = gridH_theta;
            end
            %}
            if mod(timee,10000) == 1
                
                energy_total
                energy_of_electric_field
                energy_of_magnetic_field
                energy_of_particle
                
                energy_of_particle/energy_of_electric_field
                energy_of_electric_field/energy_of_magnetic_field
                T_e = mean(m_e*c^2*double(1./sqrt(1-(electron_velocity(:,1).^2+electron_velocity(:,2).^2+electron_velocity(:,3).^2)/c^2)-1)/k_B)
                if mod(time,10) == 1
                    
                    figure
                    quiver(H_theta_R,H_theta_Z,gridE_r(:,2:101),gridE_z(1:100,:));
                    figure
                    surf(H_theta_R,H_theta_Z,sqrt(gridE_r(:,1:100).^2+gridE_z(1:100,:).^2))
                    shading interp
                    colorbar
                    colormap(jet);
                    figure
                    surf(H_theta_R,H_theta_Z,gridH_theta)
                    shading interp
                    colorbar
                    colormap(jet);
                    figure
                    plot(H_theta_R(:,1),sum(sqrt(gridE_r(:,1:100).^2+gridE_z(1:100,:).^2)')/100)
                    
                 end

                
                %[ion1_position,ion1_velocity,ion2_position,ion2_velocity,gridE,gridB_theta] = ion_electromagneticfield(ion1_position,ion1_velocity,ion2_position,ion2_velocity,gridE,gridB_theta,delta,delta_t*200,Rp,Lp,phimax,FN);
            end        
        end
        toc
        
        %[ion1_position,ion1_velocity,ion2_position,ion2_velocity] = ion_staticelectrofield_push(ion1_position,ion1_velocity,ion2_position,ion2_velocity,delta,dt_e,Rp,Lp,phimax,FN,T_e);
        
        
        %ion1_position = gather(ion1_position);
        %ion1_velocity = gather(ion1_velocity);
        %ion2_position = gather(ion2_position);
        %ion2_velocity = gather(ion2_velocity);
        
        
        %electron_position = gather(electron_position);
        %electron_velocity = gather(electron_velocity);
        %gridE = gather(gridE);
        %gridB_theta = gather(gridB_theta);
        
        x_e = electron_position(:,1);
        y_e = electron_position(:,2);
        z_e = electron_position(:,3);
        r_e = sqrt(x_e.^2+y_e.^2);
        phi_e=atan2(y_e,x_e);
        vx_e = electron_velocity(:,1);
        vy_e = electron_velocity(:,2);
        vz_e = electron_velocity(:,3);
        x_ion1 = ion1_position(:,1);
        y_ion1 = ion1_position(:,2);
        z_ion1 = ion1_position(:,3);
        vx_ion1 = ion1_velocity(:,1);
        vy_ion1 = ion1_velocity(:,2);
        vz_ion1 = ion1_velocity(:,3);
        x_ion2 = ion2_position(:,1);
        y_ion2 = ion2_position(:,2);
        z_ion2 = ion2_position(:,3);
        vx_ion2 = ion2_velocity(:,1);
        vy_ion2 = ion2_velocity(:,2);
        vz_ion2 = ion2_velocity(:,3);
        c = 3*10^8;
        v_e = sqrt(vx_e.^2+vy_e.^2+vz_e.^2);
        T = m_e*c^2*double(1./sqrt(1-v_e.^2/c^2)-1)/double(1.6E-19)/1000; %in keV;
        n_e=length(vx_e);
        
        
        ion1_position = zeros(length(x_ion1),3);
        ion1_position(:,1) = x_ion1;
        ion1_position(:,2) = y_ion1;
        ion1_position(:,3) = z_ion1;
        ion1_velocity = zeros(length(x_ion1),3);
        ion1_velocity(:,1) = vx_ion1;
        ion1_velocity(:,2) = vy_ion1;
        ion1_velocity(:,3) = vz_ion1;
        ion2_position = zeros(length(x_ion2),3);
        ion2_position(:,1) = x_ion2;
        ion2_position(:,2) = y_ion2;
        ion2_position(:,3) = z_ion2;
        ion2_velocity = zeros(length(x_ion2),3);
        ion2_velocity(:,1) = vx_ion2;
        ion2_velocity(:,2) = vy_ion2;
        ion2_velocity(:,3) = vz_ion2;
        
        T_e = mean(m_e*c^2*double(1./sqrt(1-v_e.^2/c^2)-1)/k_B,1); %in keV;
        %{
        for timee=1:cycle_time
            [ion1_position,ion1_velocity,ion2_position,ion2_velocity,RHO,rho_e,PHI,T_e,gridE_r] = ion_staticelectrofield_push(ion1_position,ion1_velocity,ion2_position,ion2_velocity,delta,delta_t,Rp,Lp,phimax,FN,T_e,PHI,RHO);
            %histogram(sqrt(ion2_velocity(:,1).^2+ion2_velocity(:,2).^2+ion2_velocity(:,3).^2));
            %histogram(sqrt(ion2_velocity(:,1).^2+ion2_velocity(:,2).^2+ion2_velocity(:,3).^2));
        end
        surf(R,Z,RHO);
        kk = kk+1;
        fmat(kk,:) = getframe;
        
        x_ion1 = ion1_position(:,1);
        y_ion1 = ion1_position(:,2);
        z_ion1 = ion1_position(:,3);
        vx_ion1 = ion1_velocity(:,1);
        vy_ion1 = ion1_velocity(:,2);
        vz_ion1 = ion1_velocity(:,3);
        x_ion2 = ion2_position(:,1);
        y_ion2 = ion2_position(:,2);
        z_ion2 = ion2_position(:,3);
        vx_ion2 = ion2_velocity(:,1);
        vy_ion2 = ion2_velocity(:,2);
        vz_ion2 = ion2_velocity(:,3);
        %}
        %{
        length(x_ion1)+length()
        rho_e_normalized = rho_e/trapz(gridz,trapz(gridr,rho_e.*R))
        for i = 1:length(gridr)
            for j = 1:length(gridz)
                
            end
        end
        %}
    end
    %{
    c = 3*10^8;
    v_e = sqrt(vx_e.^2+vy_e.^2+vz_e.^2);
    T = m_e*c^2*double(1./sqrt(1-v_e.^2/c^2)-1)/double(1.6E-19)/1000; %in keV;
    %% 
    [sig_elastic,sig_ionization,sig_excitation]=find_CS0( t_el, t_sel, t_ion, t_exc, T );
    %T(isnan(sig_elastic)||isnan(sig_ionization)||isnan(sig_excitation)));
    dx=v_e*dt_e;
    den = mean(mean(num_mol_density));
    p_total = dx.*(sig_elastic+sig_ionization+sig_excitation).*den;
    p_elastic = dx.*den.*sig_elastic./p_total;
    p_ionization = dx.*den.*sig_ionization./p_total;
    p_excitation = dx.*den.*sig_excitation./p_total;
    rand1=rand(n_e,1);
    collision_electron = find(rand1<p_total);
    rand2=rand(length(collision_electron),1);
    elastic_electron = collision_electron(rand2<p_elastic(collision_electron));
    ionization_electron = collision_electron(rand2>p_elastic(collision_electron) & rand2<(p_elastic(collision_electron)+p_ionization(collision_electron)));
    excitation_electron = collision_electron(rand2>(p_elastic(collision_electron)+p_ionization(collision_electron)));
    
    
    excitation_electron = [excitation_electron;ionization_electron(T(ionization_electron)<=threshold)];
    ionization_electron = ionization_electron(T(ionization_electron)>threshold);
    
    [ E_p1,E_s1,theta_p,theta_s,phi_p,phi_s ] = get_ImpIonized0( T(ionization_electron) );
    E_p2 = [];
    for j = 1:length(excitation_electron)
        i = excitation_electron(j);
        [ E_p2(j),E_dep ] = get_ImpExc2( T(i),t_exc );
    end
    delete = [ionization_electron;excitation_electron];
    T(delete) = [];    
    T = [T;E_p1;E_s1;E_p2'];
    ion_num = length(ionization_electron);
    
    r_ion1 = 0.02*rand(ion_num,1).^(1/2);
    theta = 0.5*rand(ion_num,1);
    x_ion1 = [x_ion1;r_ion1.*cos(theta)];
    y_ion1 = [y_ion1;r_ion1.*sin(theta)];
    z_ion1 = [z_ion1;Lp*rand(ion_num,1)];
    vx_ion1 = [vx_ion1;randn(ion_num,1)/sqrt(m_ion/(k_B*temperature))];
    vy_ion1 = [vy_ion1;randn(ion_num,1)/sqrt(m_ion/(k_B*temperature))];
    vz_ion1 = [vz_ion1;randn(ion_num,1)/sqrt(m_ion/(k_B*temperature))];
    
    T_e = mean(mean(T))*e/1000/length(T)/k_B
    %}
    %% check the electron number
    
    fprintf('length of r=%d,phi=%d,z=%d\n',length(r_e),length(phi_e),length(z_e));
    [INDEX_e]=put_into_cells_symmetry(r_divide,phi_divide,z_divide,dphi,r_e,phi_e,z_e,n_e,Rp,Lp,cellnumxy,sum_cell);
    warning off;
    
    
    %% sampling the collision
    [sig_elastic,sig_ionization,sig_excitation]=find_CS0( t_el, t_sel, t_ion, t_exc, T );
    %T(isnan(sig_elastic)||isnan(sig_ionization)||isnan(sig_excitation)));
    dx=v_e*dt_e;
    p_total = dx.*(sig_elastic+sig_ionization+sig_excitation).*num_mol_density(INDEX_e);
    p_elastic = dx.*num_mol_density(INDEX_e).*sig_elastic./p_total;
    p_ionization = dx.*num_mol_density(INDEX_e).*sig_ionization./p_total;
    p_excitation = dx.*num_mol_density(INDEX_e).*sig_excitation./p_total;
    rand1=rand(n_e,1);
    collision_electron = find(rand1<p_total);
    rand2=rand(length(collision_electron),1);
    elastic_electron = collision_electron(rand2<p_elastic(collision_electron));
    ionization_electron = collision_electron(rand2>p_elastic(collision_electron) & rand2<(p_elastic(collision_electron)+p_ionization(collision_electron)));
    excitation_electron = collision_electron(rand2>(p_elastic(collision_electron)+p_ionization(collision_electron)));
    
    %% first: elastic
    T_e = mean(m_e*c^2*double(1./sqrt(1-v_e.^2/c^2)-1)/k_B)
    fprintf('elastic');
    length(elastic_electron)
    if length(elastic_electron)>0
        tic
        [ theta1, phi1 ] = get_ElasticSc1( t_el, t_sel_1, t_del, t_sdel, T(elastic_electron) );
        toc
    end
    [ vx_e(elastic_electron),vy_e(elastic_electron),vz_e(elastic_electron) ] = rotate0(vx_e(elastic_electron),vy_e(elastic_electron),vz_e(elastic_electron),theta1,phi1 );
    
    %% second:ionization
    T_e = mean(m_e*c^2*double(1./sqrt(1-v_e.^2/c^2)-1)/k_B)
    fprintf('ionization');
    tic
    length(ionization_electron)
    excitation_electron = [excitation_electron;ionization_electron(T(ionization_electron)<=threshold)];
    ionization_electron = ionization_electron(T(ionization_electron)>threshold);

    
    x_ion1 = [x_ion1;x_e(ionization_electron)];
    y_ion1 = [y_ion1;y_e(ionization_electron)];
    z_ion1 = [z_ion1;z_e(ionization_electron)];
    vx_ion1 = [vx_ion1;randn(length(ionization_electron),1)/sqrt(m_ion/(k_B*temperature))];
    vy_ion1 = [vy_ion1;randn(length(ionization_electron),1)/sqrt(m_ion/(k_B*temperature))];
    vz_ion1 = [vz_ion1;randn(length(ionization_electron),1)/sqrt(m_ion/(k_B*temperature))];
    n = length(T);
    for j = 1:length(ionization_electron)
        i = ionization_electron(j);
        [ E_p,E_s,theta_p,theta_s,phi_p,phi_s ] = get_ImpIonized( T(i) );
        
        v_p=findV(E_p*1000*1.6E-19);
        ratio=v_p/v_e(i);
        if ratio>1
            disp(' e transport ionization error');
            %fprintf('EP=%f,T=%f\n',E_p,T(i));
        end
        vx_e(i)=ratio*vx_e(i);
        vy_e(i)=ratio*vy_e(i);
        vz_e(i)=ratio*vz_e(i);
        [ vx_e(i),vy_e(i),vz_e(i) ] = rotate(vx_e(i),vy_e(i),vz_e(i),theta_p,phi_p );
        v_e(i)=v_p;
        T(i)=E_p;
        
        
        %secondary e
        n_e=n_e+1;
        x_e(n_e,1)=x_e(i);
        y_e(n_e,1)=y_e(i);
        z_e(n_e,1)=z_e(i);
        r_e(n_e,1)=r_e(i);
        phi_e(n_e,1)=phi_e(i);
        v_s=findV(E_s*1000*1.6E-19);
        vx_e(n_e,1)=0;
        vy_e(n_e,1)=0;
        vz_e(n_e,1)=v_s;
        v_e(n_e,1)=v_s;
        [ vx_e(n_e),vy_e(n_e),vz_e(n_e) ] = rotate(vx_e(n_e),vy_e(n_e),vz_e(n_e),theta_s,phi_s );
        T(n_e)=E_s;
    end
    toc
    %% third:excitation
    T_e = mean(m_e*c^2*double(1./sqrt(1-v_e.^2/c^2)-1)/k_B)
    
    fprintf('excitation');
    length(excitation_electron)
    tic
    for j = 1:length(excitation_electron)
        i = excitation_electron(j);
        [ E_p,E_dep ] = get_ImpExc2( T(i),t_exc );
        v_p=findV(E_p*1000*1.6E-19);
        ratio=v_p/v_e(i);
        if ratio>1
            disp(' e transport ionization error');
        end
        vx_e(i)=ratio*vx_e(i);
        vy_e(i)=ratio*vy_e(i);
        vz_e(i)=ratio*vz_e(i);
        v_e(i)=v_p;
        T(i)=E_p;
    end
    %}
    toc
    T_e = mean(m_e*c^2*double(1./sqrt(1-v_e.^2/c^2)-1)/k_B)
end
%%%%%%%%plot%%%%%%%%%%%
%{
electron_position_r_abs = sqrt(electron_position(:,1).^2+electron_position(:,2).^2);
electron_cos_theta = electron_position(:,1)./electron_position_r_abs;
electron_sin_theta = electron_position(:,2)./electron_position_r_abs;
v_r = electron_velocity(:,1).*electron_cos_theta + electron_velocity(:,2).*electron_sin_theta;
histogram(v_r)
ion1_r_abs = sqrt(ion1_position(:,1).^2+ion1_position(:,2).^2);
ion2_r_abs = sqrt(ion2_position(:,1).^2+ion2_position(:,2).^2);
electron_position_r_abs = sqrt(electron_position(:,1).^2+electron_position(:,2).^2);

rho_ion2=zeros(length(gridr),1);
rho_ion1=zeros(length(gridr),1);
rho_electron=zeros(length(gridr),1);
rho = zeros(length(gridr),1);
parfor i=1:length(gridr)
     rho_ion2(i) =  FN*length(find((ion2_r_abs<i*delta_r)&(ion2_r_abs>(i-1)*delta_r)));%/(phimax*((i*delta_r)^2-((i-1)*delta_r)^2)*Lp);
     rho_ion1(i) =  FN*length(find((ion1_r_abs<i*delta_r)&(ion1_r_abs>(i-1)*delta_r)));%/(phimax*((i*delta_r)^2-((i-1)*delta_r)^2)*Lp);
     rho_electron(i) =  FN*length(find((electron_position_r_abs<i*delta_r)&(electron_position_r_abs>(i-1)*delta_r)));%/(phimax*((i*delta_r)^2-((i-1)*delta_r)^2)*Lp);
        rho(i) = 2*rho_ion2(i)+rho_ion1(i)-rho_electron(i);
end

plot(gridr,rho);
%}

end

