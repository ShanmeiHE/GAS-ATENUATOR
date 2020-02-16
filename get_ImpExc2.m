function [ E_p,E_dep ] = get_ImpExc( E_PE,t_exc )
% Electrons in the atom get impact excited 

% Input:
% E_PE: energy of the incident photoelectron (unit:keV)
% Output:
% E_p : energy of the incident electron after impact excitation in keV;
% E_dep : the mean energy deposited to the molecule in keV;


q0 = 6.514*10^(-14); % unit: (eV*cm)^2
T = E_PE * 1000; % convert to eV
k = t_exc(:,1);
A = t_exc(:,2);
omega = t_exc(:,3);
v = t_exc(:,4);
gamma = t_exc(:,5);

k_1 = 0;
k_2 = 0;

for i = 1:length(k)
    % equation is too long, break it into a,b,c
    a = q0 * A(i)/(k(i)^2);
    b = (k(i)/T)^omega(i);
    c = (1-(k(i)/T)^gamma(i))^v(i);
    % sum up
    k_1 = k_1 + a*b*c*k(i);
    k_2 = k_2 + a*b*c;
end 

E_dep = ( k_1 / k_2 ) / 1000; % change the unit backt into keV
E_p = E_PE - E_dep; 

end

