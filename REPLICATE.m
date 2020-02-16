function [r,z,phi,vx,vy,vz] = REPLICATE(r,z,phi,vx,vy,vz,N_total,amplification,r_divide,z_divide,phi_divide,dr,dphi,dz,Lp,Rp,phimax,cellnumxy)
%% check
%{
n = 1000;
N_total = n;
vx = rand(n,1);
vy = rand(n,1);
vz = rand(n,1);
amplification = 8;
r_divide = 15;
z_divide = 15;
phi_divide =[ 1;1;1;2;2;3;3;4;4;5;5;6;6;7;7];
dphi = phimax./phi_divide;
dr = Rp/r_divide;
dz = Lp/z_divide;
Lp = 0.0100;
phimax = 0.5000;
Rp = 0.0100;
cellnumxy = 57;
r = Rp*rand(n,1).^0.5;
phi = phimax*rand(n,1);
z = Lp*rand(n,1);
x = r.*cos(phi);
y = r.*sin(phi);
figure
scatter3(x,y,z,25,'filled')
%}
%% function

ddr = dr/amplification;
ddz = dz/amplification;
repeat_position_phi=zeros(N_total,amplification);
for j = 1:r_divide
    r_min = (j-1)*dr;
    r_max = j*dr;
    for i = 1:phi_divide(j)
        phi_min = (i-1)*dphi(j);
        phi_max = i*dphi(j);
        ddphi = dphi(j)/amplification;
        a = find(phi>phi_min & phi<phi_max & r>r_min & r<r_max);
        if length(a)>0
            repeat_phi = repmat(phi(a),1,amplification)';
            l=1:amplification;
            replicate_add = [repmat(l,length(a),1)*ddphi]';           
            replicate_phi = repeat_phi+replicate_add;
            b = find(replicate_phi>phi_max);
            replicate_phi(b) = replicate_phi(b)-dphi(j);
            repeat_position_phi(a,:) = [replicate_phi]';
        end
    end
end

phi = reshape(repeat_position_phi',N_total*amplification,1);
repeat_r = repmat(r,1,amplification)';
r = reshape(repeat_r,length(repeat_r(:,1))*length(repeat_r(1,:)),1);
repeat_z = repmat(z,1,amplification)';
z = reshape(repeat_z,length(repeat_z(:,1))*length(repeat_z(1,:)),1);
repeat_vx = repmat(vx,1,amplification)';
vx = reshape(repeat_vx,length(repeat_vx(:,1))*length(repeat_vx(1,:)),1);
repeat_vy = repmat(vy,1,amplification)';
vy = reshape(repeat_vy,length(repeat_vy(:,1))*length(repeat_vy(1,:)),1);
repeat_vz = repmat(vz,1,amplification)';
vz = reshape(repeat_vz,length(repeat_vz(:,1))*length(repeat_vz(1,:)),1);




N_total = N_total*amplification;

repeat_position_r=zeros(N_total,amplification);
for i = 1:r_divide
    r_min = (i-1)*dr;
    r_max = i*dr;
    a = find(r>r_min & r<r_max);
    if length(a)>0
        repeat_r = repmat(r(a),1,amplification)';
        l=1:amplification;
        replicate_add = [repmat(l,length(a),1)*ddr]';
        replicate_r = repeat_r+replicate_add;
        b = find(replicate_r>r_max);
        replicate_r(b) = replicate_r(b)-dr;
        repeat_position_r(a,:) = [replicate_r]';
    end
end

r = reshape(repeat_position_r',N_total*amplification,1);

repeat_z = repmat(z,1,amplification)';
z = reshape(repeat_z,length(repeat_z(:,1))*length(repeat_z(1,:)),1);
repeat_phi = repmat(phi,1,amplification)';
phi = reshape(repeat_phi,length(repeat_phi(:,1))*length(repeat_phi(1,:)),1);
repeat_vx = repmat(vx,1,amplification)';
vx = reshape(repeat_vx,length(repeat_vx(:,1))*length(repeat_vx(1,:)),1);
repeat_vy = repmat(vy,1,amplification)';
vy = reshape(repeat_vy,length(repeat_vy(:,1))*length(repeat_vy(1,:)),1);
repeat_vz = repmat(vz,1,amplification)';
vz = reshape(repeat_vz,length(repeat_vz(:,1))*length(repeat_vz(1,:)),1);


N_total = N_total*amplification;





repeat_position_z=zeros(N_total,amplification);
for i = 1:z_divide
    
    z_min = (i-1)*dz;
    z_max = i*dz;
    a = find(z>z_min & z<z_max);
    if length(a)>0
        repeat_z = repmat(z(a),1,amplification)';
        l=1:amplification;
        replicate_add = [repmat(l,length(a),1)*ddz]';
        replicate_z = repeat_z+replicate_add;
        b = find(replicate_z>z_max);
        replicate_z(b) = replicate_z(b)-dz;
        repeat_position_z(a,:) = [replicate_z]';
    end
    
end

z = reshape(repeat_position_z',N_total*amplification,1);
repeat_r = repmat(r,1,amplification)';
r = reshape(repeat_r,length(repeat_r(:,1))*length(repeat_r(1,:)),1);
repeat_phi = repmat(phi,1,amplification)';
phi = reshape(repeat_phi,length(repeat_phi(:,1))*length(repeat_phi(1,:)),1);

repeat_vx = repmat(vx,1,amplification)';
vx = reshape(repeat_vx,length(repeat_vx(:,1))*length(repeat_vx(1,:)),1);
repeat_vy = repmat(vy,1,amplification)';
vy = reshape(repeat_vy,length(repeat_vy(:,1))*length(repeat_vy(1,:)),1);
repeat_vz = repmat(vz,1,amplification)';
vz = reshape(repeat_vz,length(repeat_vz(:,1))*length(repeat_vz(1,:)),1);


