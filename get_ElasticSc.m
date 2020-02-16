function [ theta_ES, chi_ES ] = get_ElasticSc( t_el, t_del, E_PE )
% Electron get elastic scattered 
% Sample the deflected polar angle and the azimuthal angle;

% Input:
% E_PE: energy of photoelectron in keV
% t_el: table of data of elastic scattering TCS for energy > 50eV; 
% t_del: table of data of elastic scattering DCS for energy > 50eV; 


% Output:
% theta_ES / chi_ES: deflected polar / azimuthal angle due to elastic scattering

% gas = "Nitrogen";
% E_PE = 1;












% plot data points of DCS
[len,width] = size(t_del);
angle = [0:1:180]';
E = t_del(1,2:width);
DCS = t_del(2:len,2:width);
X = log(E);
Y = angle;
V = log(DCS);
% MeshDCS = mesh(X, Y, V);
% plot data points of TCS
x = E;
y = t_el(2,:);
% plotTCS = plot(x,y);


% Find DCS VS THETA of specified energy
x_1 = log(E_PE);
y_1 = [0:180];

A = griddata(X,Y,V,x_1,y_1);
DCS_A = exp(A);
% change unit and times 2 for nitrogen molecules
DCS_A = 2*pi * sind(angle) .* DCS_A *2 ;
% Find TCS of specified energy and times 2 for nitrogen molecules
TCS_A = interp1(x,y,E_PE) * 2;

% Construct PDF
PDF = 1/TCS_A * DCS_A;
A = [0:pi/180:pi]';
PDF_fit = fit(A,PDF,'linearinterp');

% Bisection method to find theta
TOL = 10^(-4); % set tolarance
N0 = 1000; % set maximum number of iteration

i = 1;
a = 0;
b = pi;
Rn = rand(1);
FA = Rn;

while i<=N0
    p = a + (b-a)/2; % find initial trial value
    f = integrate(PDF_fit,0,p);
    f = abs(f);
    FP = Rn-f;
    
    %   check tolarance
    if abs(FP) <= TOL
        break
    end
    
    i=i+1;
    
    if FP * FA > 0
        a = p;
        FA = FP;
    else
        b = p;
    end
end

theta_ES = p;
chi_ES = 2*pi*rand;

end


