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

sigma_el = zeros(length(E_PE),1,'gpuArray');
%% Elastic: 
a = find(E_PE >= 0.05);
    sqr_a0 = 2.8002852E-21; % unit: m^2
    x_el = t_el(1,:);
    y_el = t_el(2,:);
    
    sigma_el(a) = interp1(x_el,y_el,E_PE(a)) * sqr_a0 * 2; % times 2 for nitrogen molecules
b = find (E_PE < 0.05);
    x_el = t_sel(:,1);
    y_el = t_sel(:,2);
    sigma_el(b) = interp1(x_el,y_el,E_PE(b)*1000) * 10^(-20);
    sigma_el = sigma_el';
    
%% Electron impact ionization:
sqr_A0 = 10^(-20); % unit: m^2f
T = E_PE * 1000; % convert to eV
x_ion = t_ion(:,1);
y_ion = t_ion(:,2);
sigma_ion = interp1(x_ion,y_ion,T) * sqr_A0; % of nitrogen molecules
 
%% Electron impact excitation:
% Fitting parameter for Nitrogen:
k = t_exc(:,1);
A_2 = t_exc(:,2);
omega_2 = t_exc(:,3);
v_2 = t_exc(:,4);
gamma_2 = t_exc(:,5);
q0 = 6.542 * 10^(-14);




sigma_exc = 0;
for i = 1:length(k)
    % equation is too long, break it into a,b,c
    a = q0 * A_2(i)/(k(i)^2);
    b = (k(i)./T).^omega_2(i);
    c = (1-(k(i)./T).^gamma_2(i)).^v_2(i);
    % sum up
    sigma_exc = sigma_exc + a.*b.*c;
end 
sigma_exc = sigma_exc * 10^(-4); % change unit into m^2



sigma_el = sigma_el';
end

