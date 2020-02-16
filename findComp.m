function [CE_angle,CS_angle,E_Comp] =findComp(Ex,length)
% input: 
% length denotes number of electron/photon
% Ex is the energy of incident photon in keV
% output:
% E_Comp is the energy transfer in Joule
% CS_angle is the scattered angle of photon
% CE_angle is the scattered angle of electron

% scattering angle of photon; Klein-Nishina distribution
CS_angle = findCSangle( length ,Ex ); 

% energy of scattered photon in keV
m_e = 511; %rest mass in keV
Ex_prime = Ex ./ (1+(Ex/m_e)*(1-cos (CS_angle)));

% energy of electron in Joule
E_Comp = Ex - Ex_prime; % keV
E_Comp = E_Comp * 1.6 * 10^(-16); 

% scattering angle of electron
v = (Ex-Ex_prime.*cos(CS_angle))./sqrt(Ex.^2+Ex_prime.^2 - 2.*Ex.*Ex_prime.*cos(CS_angle));
CE_angle = acos (v);

end

