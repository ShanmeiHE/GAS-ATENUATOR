                                                                                                                                                                                                                                                                                                                                                                                                                                                                      %(* ::Package:: *)

%% update 8.25 
% input parameters:real physical parameters % unit is SI
kb=1.38064853E-23;  
T0=300; 
Rp=1E-2; % radius of the attenuator
Lp=0.03;  % length of the attenuator
phimax=0.5; % the degree phi 's region is from 0 to phimax
V=0.5*Rp^2*phimax*Lp;
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

% parameters for laser
laser_r=0.003;
Ex=1;  % energy of each photon :1keV
m_a=14; % N2
No_photon_initial = 1.25E16*phimax/2/pi;
% I0=1.6E-4; % energy of one pulse

% parameters for electron transport
dt_e=1*1E-10;


% input parameters:simulation parameters
SCmax=15.0*sigmaref*sqrt(kb*T0/m); % max of cr*sigma
alpha_=1; % parameter in VSS alpha=1 is VHS
r_divide=floor(Rp/lambda)+1;  % num to divide r
z_divide=floor(Lp/lambda)+1; % num to divide z
dr=Rp/r_divide;
dz=Lp/z_divide; % step of z
dphi=zeros(r_divide,1);
[cellnumxy,dphi,phi_divide,sum_cell] = cell_structure( phimax,Rp,r_divide,lambda);
cellnum=cellnumxy*z_divide;
dt=0.2*1E-6; % step of time
N_total=floor(int64(5000*cellnum)); % number of simulated mol.
N_total0=N_total;
FN=double (N_real)/double(N_total); % the number of real mol. represented by simulated mol.
N1=3*N_total/cellnum; % original: 3  used for collision process --parfor
% t=0;  % to record the whole time
N_collision=0;
vtheory=num_density*sqrt(2)*sqrt(8*kb*T0/pi/m)*pi*d^2;

% parameter for counting temperature or num density
n_r=10;
n_z=1;
Tfluc=0;
numfluc=0;

% produce randomly distributed simulated molecules:position
r=Rp*random_number (N_total,289108).^0.5;
z=random_number (N_total,89177451).*Lp;
phi=random_number (N_total,3492).*phimax;
INDEX=zeros(N_total,1);
x=r.*cos(phi);
y=r.*sin(phi);
figure
scatter3(x,y,z,1,'filled')
% axis equal

% produce randomly distributed simulated molecules:velocity
sig=sqrt(kb*T0/m);
voriginal=normrnd(0,sig,[N_total,3]); % velocity follows gauss distribution
vx=voriginal(:,1);
vy=voriginal(:,2);
vz=voriginal(:,3);
v=sqrt(vx.^2+vy.^2+vz.^2);
vbar=mean(v);
% hist(v,30);
% scatter3(z,x,y,10,v,'filled'); alpha 0.3;
%  [INDEX,VC] = put_into _cells _symmetry(r_divide,phi_divide,z_divide,dphi,r,phi,z,N_total,Rp,Lp,cellnumxy,sum_cell);
% return;
% electron matrix
x_e=[];
y_e=[];
z_e=[];
x_ion1=[];
y_ion1=[];
z_ion1=[];
x_ion2=[];
y_ion2=[];
z_ion2=[];
r_e=[];
phi_e=[];
vx_e=[];
vy_e=[];
vz_e=[];
vx_ion1=[];
vy_ion1=[];
vz_ion1=[];
vx_ion2=[];
vy_ion2=[];
vz_ion2=[];
v_e=[];   
% internal energy
xi=rand(N_total,1);
ei=-kb*T0*log(xi);

% parameters for electricmagnetic field
delta = 1*10^(-4);
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


PML_length = delta_z*10;
hole_radius = delta_r*50;
%% PML z<0
H_theta_gridr_PML0 = delta_r/2:delta_r:hole_radius+delta_r/2;
H_theta_gridz_PML0 = -delta_z/2-PML_length:delta_z:-delta_z/2;

H_theta_lenr_PML0 = length(H_theta_gridr_PML0);
H_theta_lenz_PML0 = length(H_theta_gridz_PML0);




E_r_gridr_PML0 = delta_r/2:delta_r:hole_radius+delta_r/2;
E_r_gridz_PML0 = -PML_length:delta_z:-delta_z;

E_r_lenr_PML0 = length(E_r_gridr_PML0);
E_r_lenz_PML0 = length(E_r_gridz_PML0);





E_z_gridr_PML0 = 0:delta_r:hole_radius;
E_z_gridz_PML0 = -delta_z/2-PML_length:delta_z:-delta_z/2;

E_z_lenr_PML0 = length(E_z_gridr_PML0);
E_z_lenz_PML0 = length(E_z_gridz_PML0);
%% PML z>n
H_theta_gridr_PMLn = delta_r/2:delta_r:hole_radius+delta_r/2;
H_theta_gridz_PMLn = z+delta_z/2:delta_z:z+delta_z/2+PML_length;

H_theta_lenr_PMLn = length(H_theta_gridr_PMLn);
H_theta_lenz_PMLn = length(H_theta_gridz_PMLn);




E_r_gridr_PMLn = delta_r/2:delta_r:hole_radius+delta_r/2;
E_r_gridz_PMLn = z+delta_z:delta_z:z+delta_z+PML_length;

E_r_lenr_PMLn = length(E_r_gridr_PMLn);
E_r_lenz_PMLn = length(E_r_gridz_PMLn);





E_z_gridr_PMLn = 0:delta_r:hole_radius;
E_z_gridz_PMLn = z+delta_z/2:delta_z:z+delta_z/2+PML_length;

E_z_lenr_PMLn = length(E_z_gridr_PMLn);
E_z_lenz_PMLn = length(E_z_gridz_PMLn);



gridE_r = zeros(E_r_lenr,E_r_lenz);
gridE_z = zeros(E_z_lenr,E_z_lenz);
gridH_theta = zeros(H_theta_lenr,H_theta_lenz);

gridE_r_PML0 = zeros(E_r_lenr_PML0,E_r_lenz_PML0);
gridE_z_PML0 = zeros(E_z_lenr_PML0,E_z_lenz_PML0);
gridD_r_PML0 = zeros(E_r_lenr_PML0,E_r_lenz_PML0);
gridD_z_PML0 = zeros(E_z_lenr_PML0,E_z_lenz_PML0);
gridH_theta_PML0 = zeros(H_theta_lenr_PML0,H_theta_lenz_PML0);

gridE_r_PMLn = zeros(E_r_lenr_PMLn,E_r_lenz_PMLn);
gridE_z_PMLn = zeros(E_z_lenr_PMLn,E_z_lenz_PMLn);
gridD_r_PMLn = zeros(E_r_lenr_PMLn,E_r_lenz_PMLn);
gridD_z_PMLn = zeros(E_z_lenr_PMLn,E_z_lenz_PMLn);
gridH_theta_PMLn = zeros(H_theta_lenr_PMLn,H_theta_lenz_PMLn);




gridr = 0:delta:Rp;
gridz = 0:delta_z:Lp;
len=length(gridr)*length(gridz);
[R,Z] = meshgrid(gridr,gridz);
R = R';
Z = Z';
PHI = zeros(length(gridr),length(gridz));
RHO = zeros(length(gridr),length(gridz));

% file
% fileID=fopen('dt_ 0.2_collision _frequecy.numeachcell.80.time100.txt','wt');
%   tem=fopen('temperature_to _time.symmetry.txt','wt');
%   num=fopen('numdensity_to _time.symmetry.txt','wt');
% attenuation_coefficient=fopen('attenuation_coefficient _to _time.symmetry.time100.txt','wt');
% avetem=fopen('temperaturebar_to _time.symmetry.txt','wt');
% tem_r=fopen('temperature_to _r _t0.symmetry.txt','wt');




%% start time loop
for TIME=1:100
    TIME
    %% go on with new v during the time period of dt
    [x,y,z,r,phi] = free_motion( vx,vy,vz,x,y,z,dt );


    %% rebound
    [x,y,z,r,phi,vx,vy,vz,v,N_boundary] = boundary_conditions_specular( x,y,z,r,phi,vx,vy,vz,v,dt,Rp,Lp,phimax,T0,m);


    %% put the molecules into cells
    [INDEX] = put_into_cells_symmetry(r_divide,phi_divide,z_divide,dphi,r,phi,z,N_total,Rp,Lp,cellnumxy,sum_cell);
    VC=calculate_VC( Rp,Lp,r_divide,phi_divide,z_divide,dphi,cellnumxy,sum_cell,cellnum );


     %% calculate the collision process
    [ vx,vy,vz,v,N_collision,ncollisiontime,ei,n_SC ] = collision_HS_internal( INDEX,N1,cellnum,vx,vy,vz,v, ...
         SCmax,FN,VC,dt,d,alpha_,N_collision,ei,w,zeta,m);
    % [ vxout,vyout,vzout,vout,N_collisionout,ncollisiontime ] = collision_VHS _VSS( INDEX,N1,cellnum, ...
    %     vx,vy,vz,v,SCmax,FN,VC,dt,d,alpha_,N_collision,m,kb,sigmaref,Tref,w);


    %% put the molecules into cells
    [INDEX] = put_into_cells_symmetry(r_divide,phi_divide,z_divide,dphi,r,phi,z,N_total,Rp,Lp,cellnumxy,sum_cell);
    VC=calculate_VC( Rp,Lp,r_divide,phi_divide,z_divide,dphi,cellnumxy,sum_cell,cellnum );

    %% laser input
    if mod(TIME,5)==1
        
    %% change the statistical weight
    
    phi = atan2(y,x);
    r = sqrt(x.^2+y.^2);
    dr = Rp/r_divide;
    dz = Lp/z_divide;
    amplification = 4;

    fprintf('REPLICATE time:\n')
    tic
    [r,z,phi,vx,vy,vz] = REPLICATE(r,z,phi,vx,vy,vz,N_total,amplification,r_divide,z_divide,phi_divide,dr,dphi,dz,Lp,Rp,phimax);
    toc
    x = r.*cos(phi);
    y = r.*sin(phi);
    v = sqrt(vx.^2+vy.^2+vz.^2);
    FN = FN/amplification^3;
    %}
    
    %% laser input
    position=[x y z r]; 
    velocity=[vx vy vz v];
    [p] = GUASS(delta_r,laser_r);


    fprintf('INCIDENCE time:\n')
    tic
    [ No_photon,E_position,E_velocity, E_theta, E_chi, Ion_position, Ion_velocity,charge ] = INCIDENCE0(laser_r, ...
        Lp,z_divide,m_a,FN, Ex, No_photon_initial,position,velocity,phimax,N_total,p,delta_r);
    toc
    No_photon
    x_e = [x_e;E_position(:,1)];
    y_e = [y_e;E_position(:,2)];
    z_e = [z_e;E_position(:,3)];
    r_e = [r_e;E_position(:,4)];
    phi_e = [phi_e;atan2(y_e,x_e)];
    vx_e = [vx_e;E_velocity(:,1)];
    vy_e = [vy_e;E_velocity(:,2)];
    vz_e = [vz_e;E_velocity(:,3)];
    v_e = [v_e;E_velocity(:,4)];

    x_ion2 = [x_ion2;Ion_position(:,1)];
    y_ion2 = [y_ion2;Ion_position(:,2)];
    z_ion2 = [z_ion2;Ion_position(:,3)];
    vx_ion2 = [vx_ion2;Ion_velocity(:,1)];
    vy_ion2 = [vy_ion2;Ion_velocity(:,2)];
    vz_ion2 = [vz_ion2;Ion_velocity(:,3)];
    n_e = length(x_e)

    %% plot electron density
    n_e = length(x_e);
    r_e = sqrt(x_e.^2+y_e.^2);
    rho_electron=zeros(length(gridr),1);
    parfor i=1:length(gridr)
         rho_electron(i) =  FN*length(find((r_e<i*delta_r)&(r_e>(i-1)*delta_r)))/(phimax*((i*delta_r)^2-((i-1)*delta_r)^2)*Lp);
    end
    figure
    plot(gridr,rho_electron);


    %% electron transport    
    [mol_rho] = particle_density(dr,dz,dphi,r,phi,z,cellnumxy,sum_cell,VC,FN);
    FN = FN/10^3;
    [ gridE_r,gridE_z,gridH_theta,gridE_r_PML0,gridE_z_PML0,gridD_r_PML0,gridD_z_PML0,gridH_theta_PML0,gridE_r_PMLn,gridE_z_PMLn,gridD_r_PMLn,gridD_z_PMLn,gridH_theta_PMLn,x_e,y_e,z_e,vx_e,vy_e,vz_e,x_ion1,y_ion1,z_ion1,vx_ion1,vy_ion1,vz_ion1,x_ion2,y_ion2,z_ion2,vx_ion2,vy_ion2,vz_ion2,PHI,RHO] = electron_transport_field2( x,y,z,vx,vy,vz,v,charge,INDEX,VC,FN,x_e,y_e,z_e,r_e,phi_e,vx_e,vy_e,vz_e,v_e,x_ion1,y_ion1,z_ion1,vx_ion1,vy_ion1,vz_ion1,x_ion2,y_ion2,z_ion2,vx_ion2,vy_ion2,vz_ion2,n_e,dt_e,dt,m,m_e,r_divide,phi_divide,z_divide,dphi,Rp,Lp,cellnumxy,sum_cell,phimax,gridE_r,gridE_z,gridH_theta,gridE_r_PML0,gridE_z_PML0,gridD_r_PML0,gridD_z_PML0,gridH_theta_PML0,gridE_r_PMLn,gridE_z_PMLn,gridD_r_PMLn,gridD_z_PMLn,gridH_theta_PMLn,mol_rho,delta,PHI,RHO);
    % fprintf(attenuation_coefficient,'% d % f \n',time-1,No_photon/No_photon_initial);  
    FN = FN*10^3;
    
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
    
    [r,z,phi,vx,vy,vz] = SamplingBack(amplification,r,z,phi,vx,vy,vz);
    FN = FN*amplification^3;
    x=r.*cos(phi);
    y=r.*sin(phi);
    figure
    scatter3(x,y,z,1,'filled')

    end
    
    

end
% movie(fmat2)
