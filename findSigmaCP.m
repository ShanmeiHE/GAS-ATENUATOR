function [ Sigma_CP ] = findSigmaCP( Ex)
% Sigma_CP : cross-section of compton scattering
% Ex : energy of X-ray (unit: keV)

% The data points has Ex unit in MeV, 
% convert input value into MeV
Ex = Ex / 1000;
N_A = 6.02 * 10^(23);
m_a = 14; % since we only consider nitrogen

T = readtable('Compton_cs.csv');
t = table2array(T);
X = t(:,1);
Y = t(:,2);
loglog(X,Y);

% Get value for mass attenuation coefficient
MAC_CS = interp1(X,Y, Ex);

% find sigmaCP in m^(-2);
Sigma_CP = MAC_CS * m_a / N_A * 10^(-4);
end
