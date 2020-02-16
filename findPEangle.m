function [ PE_angle ] = findPEangle( dE, len )
% input: energy of electron and number of electron
% output: the angle of photoelectron
% given by Sauter distribution
% define constants
m_e = 9.11*10^(-31);
c = 3*10^8;
E_rest = m_e*c^2;

% calculate parameters
gamma = 1+dE/E_rest;
beta = sqrt(dE*(dE+2*E_rest))/(dE+E_rest);
A = 1/beta - 1;
g0 = 2*(1/A+0.5*beta*gamma*(gamma-1)*(gamma-2));

% Sampling by rejection method
PE_angle = zeros (len,1);
for i = 1:len
    rej = 1;
    while rej == 1
        r1 = rand ;
        v = 2*A/((A+2)^2-4*r1)*(2*r1+(A+2)*r1^(0.5));
        g_v = (2-v)*((1/(A+v))+0.5*beta*gamma*(gamma-1)*(gamma-2));
        r2 = rand;
        
        % check rejection criteria
        if r2*g0 < g_v
            rej = 0;
        end
        
    end
    PE_angle(i) = acos(1-v);
end


end

