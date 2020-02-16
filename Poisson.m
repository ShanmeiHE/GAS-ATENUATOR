function [gridE,rho_e,RHO,PHI,T_e] = Poisson(rho_i,r,z,T_e,delta,total_energy,n_e,W)
k_B = 1.38064853E-23;  
me=9.1096*10^(-31);
c=3*10^8;
epsilon=8.854187817*10^(-12);
mu=4*pi*10^(-7);
e=1.6*10^(-19);
delta_r = delta;
delta_z = delta;

gridr = 0+delta/2:delta:r-delta/2;
gridz = 0+delta/2:delta_z:z-delta/2;
len=length(gridr)*length(gridz);
[R,Z] = meshgrid(gridr,gridz);
R = R';
Z = Z';

lenr = length(gridr);
lenz = length(gridz);


int_rho_i = trapz(gridz,trapz(gridr,rho_i.*R));
delta_total_energy = total_energy;
t = 0;
while delta_total_energy > 1/100*total_energy || t<100
    t = t+1;
    PHI = zeros(lenr,lenz);
    gridE = zeros(lenr,lenz,2);
    PHI_before = ones(lenr,lenz);
    rho_e = zeros(lenr,lenz);
    RHO = zeros(lenr,lenz);
    g = zeros(lenr,lenz);
    n_e0 = 10^15;
    b=0;
    k=0;
    tic
    while max(max(abs(PHI_before-PHI))) > 1/100*mean(mean(PHI))
        k = k+1;
        PHI_before = PHI;
        for i = 2:(lenr-1)
            for j =1:lenz
                if j ~= 1 & j~=lenz
                    rho_e(i,j) = e*n_e0*exp(e*PHI(i,j)/(k_B*T_e));
                    RHO(i,j) = (rho_i(i,j)-rho_e(i,j));
                    g(i,j) = (RHO(i,j)/epsilon + (PHI(i+1,j)+PHI(i-1,j))/delta_r^2 + 1/(R(i,j)+delta_r)*(PHI(i+1,j)-PHI(i-1,j))/(2*delta_r)+(PHI(i,j-1)+PHI(i,j+1))/delta_z^2)/(2/delta_r^2+2/delta_z^2);
                elseif j == 1
                    g(i,j) = (b + (PHI(i+1,j)+PHI(i-1,j))/delta_r^2 + 1/(R(i,j)+delta_r)*(PHI(i+1,j)-PHI(i-1,j))/(2*delta_r)+(PHI(i,lenz)+PHI(i,j+1))/delta_z^2)/(2/delta_r^2+2/delta_z^2);
                else
                    g(i,j) = (b + (PHI(i+1,j)+PHI(i-1,j))/delta_r^2 + 1/(R(i,j)+delta_r)*(PHI(i+1,j)-PHI(i-1,j))/(2*delta_r)+(PHI(i,j-1)+PHI(i,1))/delta_z^2)/(2/delta_r^2+2/delta_z^2);
                end
                %fprintf('g = %f\n b = %f\n',g(i,j),b);
            end
        end
        n_e0 = n_e0*int_rho_i/trapz(gridz,trapz(gridr,rho_e.*R));
        PHI(g>0) = g(g>0);
        PHI(1,:) = PHI(2,:);
        PHI(lenr,:) = zeros(1,lenz);
        PHI(:,1) = zeros(length(PHI(:,1)),1);
        PHI(:,lenz) = zeros(length(PHI(:,lenz)),1);
        if mod(k,10^4) == 1
            1;
        end
    end
    toc
    [E_z,E_r] = gradient(PHI,delta_z,delta_r);
    gridE(:,:,1) = -E_r;
    gridE(:,:,2) = -E_z;
    
end
    


%{
figure;
surf(gridr,gridz,rho_e);
%}
RHO = rho_i-rho_e;
