function [ theta_ES, chi_ES ] = get_ElasticSc0( t_el, t_del, T )


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
y = t_el(2,:);

% MeshDCS = mesh(X, Y, V);
% plot data points of TCS

parfor i =1:n
    [ theta_ES(i), chi_ES ]=get_ElasticSc_sub( x,y,X,Y,V,T(i) );
end
theta_ES = theta_ES';
chi_ES = 2*pi*rand(n,1);
    
    
    
    
    
    