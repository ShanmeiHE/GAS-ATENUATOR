function [ theta_ES, chi_ES ] = get_ElasticSc_sub1( x,y,X,Y,V,E_PE )

% Find DCS VS THETA of specified energy
x_1 = log(E_PE);
y_1 = [0:180];

A = griddata(X,Y,V,x_1,y_1);
DCS_A = exp(A);
% change unit and times 2 for nitrogen molecules
DCS_A = 2*pi * sind(Y) .* DCS_A *2 ;
% Find TCS of specified energy and times 2 for nitrogen molecules
TCS_A = interp1(x,y,E_PE) * 2;

% Construct PDF
PDF = 1/TCS_A * DCS_A;
A = [0:pi/180:pi]';
intPDF = cumtrapz(A, PDF);
random_number = rand(1) * intPDF(length(intPDF));
theta_ES = interp1(intPDF,A,random_number);
chi_ES = 2*pi*rand;
end


