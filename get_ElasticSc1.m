function [ theta_ES, chi_ES ] = get_ElasticSc1( t_el, t_sel_1, t_del, t_sdel, T )
%% Pre-set table 
sqr_a0 = 2.8002852E-21; % unit: m^2
% change the unit of TDC for energy < 50eV
% and change to Nitrogen atom so can combine with the other table without confusion  
t_sel_1(:,2) = t_sel_1(:,2) .* 10^-20 / sqr_a0 /2 ; 
t_sel_1(:,1) = t_sel_1(:,1)./1000; % change the energy unit to keV 
t_sel_1 = t_sel_1(1:(length(t_sel_1)-1),1:2);

% Combine the table for TCS
t_el = t_el';

t_el = [t_sel_1;t_el];

% Combine the table for DCS
[l,w] = size(t_del);
t_del_1 = t_del(1:l,2:w); 
t_del = [t_sdel,t_del_1];

%% 
n = length(T);
% plot data points of DCS
[len,width] = size(t_del);
angle = [0:1:180]';
E = t_del(1,2:width);
DCS = t_del(2:len,2:width);

X = log(E);
Y = angle;
V = log(DCS);
x = E;
y = t_el(:,2);

% MeshDCS = mesh(X, Y, V);
% plot data points of TCS

parfor i =1:n
    [ theta_ES(i), chi_ES ]=get_ElasticSc_sub1( x,y,X,Y,V,T(i) );
end
theta_ES = theta_ES';
chi_ES = 2*pi*rand(n,1);

end
    
    
    
    
    
    