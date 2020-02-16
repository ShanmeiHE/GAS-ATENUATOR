function [ sigma_el, sigma_ion, sigma_exc ] = find_CS0( t_el, t_sel, t_ion, t_exc, E_PE )
% find the cross section for each collison process
% All the table used are read outside

% Input:
% E_PE: energy of the incident photoelectron (unit:keV)
% t_el: table of data of elastic scattering TCS for energy > 50eV; 
% t_sel: table of data of elastic scattering TCS for energy < 50 eV;
% t_ion: table of data of impact ionization TCS for energy > 300 eV;
% t_exc: table of parameter of impact excitation for 100eV < energy < 5keV
% Output: sigma_
% el: elastic collision;  ion: ionization;  exc: excitation

%% Elastic: 
sqr_a0 = 2.8002852E-21; % unit: m^2
x_el = t_el(1,:);
y_el = t_el(2,:);

x_sel = t_sel(:,1);
y_sel = t_sel(:,2);

%% Electron impact ionization:
sqr_A0 = 10^(-20); % unit: m^2f
T = E_PE * 1000; % convert to eV
x_ion = t_ion(:,1);
y_ion = t_ion(:,2);
 
%% Electron impact excitation:
% Fitting parameter for Nitrogen:\
k = t_exc(:,1);
A_2 = t_exc(:,2);
omega_2 = t_exc(:,3);
v_2 = t_exc(:,4);
gamma_2 = t_exc(:,5);
q0 = 6.542 * 10^(-14);
len = length(k);

cudaFilename1 = 'find_CS0.cu';
ptxFilename1 = 'find_CS0.ptx';
kernel1 = parallel.gpu.CUDAKernel( ptxFilename1, cudaFilename1 );
kernel1.ThreadBlockSize = [1024,1,1];
kernel1.GridSize = [200,1,1];

parNum = length(E_PE);
sigma_el = zeros(parNum,1,'gpuArray');
sigma_ion = zeros(parNum,1,'gpuArray');
sigma_exc = zeros(parNum,1,'gpuArray');

[~,~,~,~,~,~,~,~,~,~,~,~,sigma_el,sigma_ion,sigma_exc] = feval(kernel1,E_PE,x_el,y_el,x_sel,y_sel,x_ion,y_ion,k,A_2,omega_2,v_2,gamma_2,sigma_el,sigma_ion,sigma_exc,sqr_A0,sqr_a0,q0,len,parNum);

end

