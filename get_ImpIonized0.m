%function [ E_p,E_s,theta_p,theta_s,chi_p,chi_s ] = get_ImpIonized0( E_PE )
function [ E_p,E_s,theta_p,theta_s,chi_p,chi_s ] =get_ImpIonized0( E_PE )
% Electron in the gas molecule undergoes impact ionization

% Input:
% E_PE: energy of the incident photoelectron (unit:keV)
% Output:
% E_p / E_s : energy of the primary / secondary electron ;
% theta_p / theta_s: deflected polar angle of priamry / secondary electron; ;
% chi_p / chi_s : azimuthal angle of primary / secondary electron ;

%% Pre-set data
% constant
c = 3*10^8;
m = 9.109 * 10^(-31); % electron rest mass
E_e = m * c^2 / (1.6022 * 10 ^ (-19)); 

% Fitting parameter of Nitrogen
I = 15.6; % Ionization threshold, unit : eV
T_a = 1000; % unit : eV
T_b = 2*I;
t_b = I;
T_s = 4.17; % unit : eV
t_s = 13.8; % unit : eV


n = length(E_PE);
% Calculated value
T = E_PE * 1000 ; % unit : eV 
T_0 = T_s - T_a./(T+T_b);
t = t_s*T./(T+t_b);
T_m = (T-I)/2;
Rn = rand(n,1);

%% Find energy
% Energy of one of the electron:
k_1 = atan((T_m-T_0)./t);
k_2 = atan(T_0./t);
k = T_0 + t .* tan(Rn.*(k_1+k_2)-k_2);
E_1 = k;
% Energy of the other electron:
E_2 = T - E_1 - I;

% Define the faster particle is the primary electron
E_p = zeros(n,1);
E_s = zeros(n,1);

E_p(E_2 > E_1) = E_2(E_2 > E_1);
E_s(E_2 > E_1) = E_1(E_2 > E_1);

E_p(E_2 <= E_1) = E_1(E_2 <= E_1);
E_s(E_2 <= E_1) = E_2(E_2 <= E_1);


E_p = E_p/1000;
E_s = E_s/1000;


%% Find Angle
% Azimuthal angle
chi_p = 2*pi*rand(n,1);
chi_s = chi_p - pi;
% Polar angle
sin_theta_p = sqrt( (k./T)./((1-k./T).*T/(2*E_e)+1) );
theta_p = asin(sin_theta_p);
(1-k./T)./(1+k/(2*E_e));
sin_theta_s = sqrt( (1-k./T)./(1+k/(2*E_e)) );
theta_s = asin(sin_theta_s);
%}

end

