function [ Sigma ] = findSigmaPE( Ex, m_a )
% Ex : energy of x-ray 
% sigma: cross-section, unit: m^-2

N_A = 6.02 * 10^(23); % Avogadro constant 

% The data points has Ex unit in MeV, convert it into keV
Ex = Ex / 1000; 
T = readtable([num2str(m_a),'_MAC.csv']);
t = table2array(T); 

% plot the graph
X = t(:,1);
Y = t(:,2);
loglog(X,Y);

% find corresponding mass attenuation coefficient (MAC)
[X,index] = unique(X); 
MAC = interp1(X,Y(index), Ex);

% find sigmaPE in m^(-2);
Sigma = MAC * m_a / N_A * 10^(-4);

end

