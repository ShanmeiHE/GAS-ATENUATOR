function [ vxout,vyout,vzout,vout,N_collisionout,ncollisiontime ] = collision_VHS_VSS( INDEX,N1,cellnum,vx,vy,vz,v,SCmax,FN,VC,dt,d,alpha_,N_collision,m,kb,sigmaref,Tref,w)
%collision process,VHS/VSS model
ncollisiontime=0;
vxn=zeros(N1,cellnum);
vyn=zeros(N1,cellnum);
vzn=zeros(N1,cellnum);
vn=zeros(N1,cellnum);
sub_change=zeros(N1,cellnum);
parfor indexnum=1:cellnum
    %in the cell with index of indexnum

    a=find(INDEX==indexnum);
    N=length(a);
    vxpart=vx(a);
    vxpart(N+1:N1)=0;
    vypart=vy(a);
    vypart(N+1:N1)=0;
    vzpart=vz(a);
    vzpart(N+1:N1)=0;
    vpart=v(a);
    vpart(N+1:N1)=0;
    if N>2
        Npair=floor(SCmax*FN*N*(N-1)/VC(indexnum)*0.5*dt);
        %fprintf('%e,%e,%d\n',SCmax,VC(indexnum),FN);
        %fprintf('N=%d,NPAIR=%d\n',N,Npair);
        N_before=0;
        while(1)
            sub1=produce_random_subscript(N); %this sub is the subscript of a; to randomly choose a mol.
            sub2=produce_random_subscript(N);
            if(sub1==sub2)
                continue;
            end

            ur=vxpart(sub1)-vxpart(sub2);
            vr=vypart(sub1)-vypart(sub2);
            wr=vzpart(sub1)-vzpart(sub2);
            cr=sqrt(ur*ur+vr*vr+wr*wr);
            %SC=cr*pi*d^2; %HS
            SC=cr*sigmaref*(2*kb*Tref/(0.5*m*cr*cr))^(w-1/2)/gamma(5/2-w); %VHS/VSS
%             if SC>SCmax
%                 SCmax=SC;
%             end
            %fprintf('%f\n',SC/SCmax);
            if SC>SCmax
                disp('error');
            end
            randnum=rand(1);
            %fprintf('%f\n',SC/SCmax);
            if randnum<SC/SCmax  %collide

                um=(vxpart(sub1)+vxpart(sub2))/2;  %v of mass center
                vm=(vypart(sub1)+vypart(sub2))/2;
                wm=(vzpart(sub1)+vzpart(sub2))/2;
              if abs(alpha_-1)<1E-6  %VHS model
                  B=2*rand(1)-1; % cosX
                  A=sqrt(1-B^2); % sinX
                  C=2*pi*rand(1); %phi
                  urf=cr*B;
                  vrf=cr*A*cos(C);
                  wrf=cr*A*sin(C);
              else   %VSS model
                  B=2*rand(1)^(1/alpha_)-1;% cosX
                  A=sqrt(1-B^2); % sinX
                  C=2*pi*rand(1); %phi
                  SC=sin(C);
                  CC=cos(C);
                  D=sqrt(vr^2+wr^2);
                  if D>1E-6
                      urf=B*ur+A*SC*D;
                      vrf=B*vr+A*(cr*wr*CC-ur*vr*SC)/D;
                      wrf=B*wr-A*(cr*vr*CC-ur*wr*SC)/D;
                  else
                      urf=B*ur;
                      vrf=A*CC*ur;
                      wrf=A*SC*ur;
                  end
              end
              vxpart(sub1)=um+1/2*urf;  %v after collision
              vxpart(sub2)=um-1/2*urf;
              vypart(sub1)=vm+1/2*vrf;
              vypart(sub2)=vm-1/2*vrf;
              vzpart(sub1)=wm+1/2*wrf;
              vzpart(sub2)=wm-1/2*wrf;
              vpart(sub1)=sqrt(vxpart(sub1)^2+vypart(sub1)^2+vzpart(sub1)^2);
              vpart(sub2)=sqrt(vxpart(sub2)^2+vypart(sub2)^2+vzpart(sub2)^2);
              N_collision=N_collision+1;
%               crs=crs+cr;
%               ncr=ncr+1;
               ncollisiontime=ncollisiontime+1;
            end
%             if(N_before==0)
%                 fprintf('%f\n',SC);
%             end
            N_before=N_before+1;
            if(N_before>=Npair)
                break;
            end
        end
    end
    a(N+1:N1)=0;
    sub_change(:,indexnum)=a;
    vxn(:,indexnum)=vxpart;
    vyn(:,indexnum)=vypart;
    vzn(:,indexnum)=vzpart;
    vn(:,indexnum)=vpart;
end
vxout=vx;
vyout=vy;
vzout=vz;
vout=v;
for indexnum=1:cellnum
    a=find(sub_change(:,indexnum)>0);
    n1=length(a);
    vxout(a)=vxn(1:n1,indexnum);
    vyout(a)=vyn(1:n1,indexnum);
    vzout(a)=vzn(1:n1,indexnum);
    vout(a)=vn(1:n1,indexnum);
end
N_collisionout=N_collision;
end

