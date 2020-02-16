//nvcc -ptx electron_transport.cu -ccbin "F:Visual Studio\VC\Tools\MSVC\14.12.25827\bin\Hostx64\x64"
#include "curand_kernel.h"
__device__ void  EM1(double *x,
                     double *y,
                     double *z,
                     double *vx,
                     double *vy,
                     double *vz,
                     double *sig_elastic,
                     double *sig_ionization,
                     double *sig_excitation,
                     double *energy,
                     double *xInterp,
                     double *yInterp,
                     double *V_interp,
                     double *density,
                     double *temperature,
                     double *dphi,
                     double *kExc,
                     double *Aexc,
                     double *omegaExc,
                     double *vExc,
                     double *gammaExc,
                     bool *secondaryParticle,
                     double *secondary_vx,
                     double *secondary_vy,
                     double *secondary_vz,
                     double *particleSamplingOut,
                     double *ion1_vx,
                     double *ion1_vy,
                     double *ion1_vz,
                     int *sum_cell0,
                     const int lenExc,
                     const int cellnumxy,
                     const double dr,
                     const double dz,
                     const double dt,
                     const int parNum) {
    int globalBlockIndex = blockIdx.x + blockIdx.y * gridDim.x;
    int localThreadIdx = threadIdx.x + blockDim.x * threadIdx.y;
    int threadsPerBlock = blockDim.x*blockDim.y;
    int n = localThreadIdx + globalBlockIndex*threadsPerBlock;
    if ( n >= parNum ){
        return;
    }
    double e,c,m,pi,E_e,m_ion,kb;
    e = 1.6022e-19;
    c = 3e+08;
    m = 9.109e-31;
    m_ion = 4.65e-26;  // mass of one N2 
    kb = 1.38064853e-23;
    E_e = m*c*c / e;
    pi = 3.1415926;
    double p_elastic,p_ionization,p_excitation,p_total,a,b,p1,p2,rand,a0,b0,vInterp,E_p,v_p,ratio;
    double r,phi,v_e,v,dx,vx0,vy0,vz0,theta_ES,phi_ES,DCS_A,PDF,intPDFmax,intPDF1,intPDF2,gamma,T;
    int i,k,index_r,index_z,index_phi,index;

    double I,T_0,T_a,T_b,t_b,T_s,t_s,t,T_m,Rn,k_1,k_2,k0,v_s;
    double E_1,E_2,E_s,phi_p,phi_s,sin_theta_p,theta_p,sin_theta_s,theta_s;
    I = 15.6; //Ionization threshold, unit:eV

    double q0,aExc,bExc,cExc,E_dep;

    v_e = sqrt( vx[n]*vx[n] + vy[n]*vy[n] + vz[n]*vz[n] );
    dx = v_e*dt;

    r = sqrt(x[n]*x[n]+y[n]*y[n]);
    phi = atan(y[n]/x[n]);
    index_z = floor(z[n]/dz);
    index_r = floor(r/dr);
    index_phi = floor(phi/dphi[index_r]);
    index = index_z*cellnumxy + sum_cell0[index_r] + index_phi;

    p_elastic = dx * density[index] * sig_elastic[n];
    p_ionization = dx * density[index] * sig_ionization[n];
    p_excitation = dx * density[index] * sig_excitation[n];
    p_total = p_elastic + p_ionization + p_excitation;
    a = p_elastic/p_total;
    b = a + p_ionization/p_total;
    curandState state;
    curand_init((unsigned long long)clock(),0,n, & state);
    p1=curand_uniform_double(&state); 
    p2=curand_uniform_double(&state); 
    
    if(p1< p_total){
        if(p2 < a){
            //elastic
            i = 0;
            while( energy[n] > xInterp[i+1] ){
                i++;
            }
            a0 = (xInterp[i+1]-energy[n])/(xInterp[i+1]-xInterp[i]);
            b0 = (energy[n]-xInterp[i])/(xInterp[i+1]-xInterp[i]);
            double TCS_A = (a0*yInterp[i]+b0*yInterp[i+1])*2;
            intPDFmax = 0;
            for( k = 0; k<=180; k++ ){
                vInterp = a0*V_interp[i*181+k] + b0*V_interp[(i+1)*181+k];
                DCS_A = exp(vInterp);
                DCS_A = 2*pi*sin(k*pi/180)*DCS_A*2;
                PDF = 1/TCS_A * DCS_A;
                intPDFmax = intPDFmax + PDF* pi/180;
            }
            rand = curand_uniform_double(&state) * intPDFmax;
            intPDF1 = 0;
            k = 0;
            while( rand > intPDF1 && k<181 ){
                vInterp = a0*V_interp[i*181+k] + b0*V_interp[(i+1)*181+k];
                DCS_A = exp(vInterp);
                DCS_A = 2*pi*sin(k*pi/180)*DCS_A*2;
                PDF = 1/TCS_A * DCS_A;
                intPDF1 = intPDF1 + PDF* pi/180;
                k++;
            }
            vInterp = a0*V_interp[i*181+k] + b0*V_interp[(i+1)*181+k];
            DCS_A = exp(vInterp);
            DCS_A = 2*pi*sin(k*pi/180)*DCS_A*2;
            PDF = 1/TCS_A * DCS_A;
            intPDF2 = intPDF1 + PDF* pi/180;
            a0 = (intPDF2 - rand)/(intPDF2 - intPDF1);
            b0 = (rand - intPDF1)/(intPDF2 - intPDF1);
            theta_ES = ( a0*(k-1)+b0*k )*pi/180;
            phi_ES = 2* pi* curand_uniform_double(&state);

            v = sqrt( vx[n]*vx[n] + vy[n]*vy[n] + vz[n]*vz[n] );
            if( v*v - vz[n]*vz[n] > 0.0001){
                vx0 = vx[n]*cos(theta_ES) + sin(theta_ES)/sqrt(v*v-vz[n]*vz[n])*( vx[n]*vz[n]*cos(phi_ES) - v*vy[n]* sin(phi_ES) );
                vy0 = vy[n]*cos(theta_ES) + sin(theta_ES)/sqrt(v*v-vz[n]*vz[n])*( vy[n]*vz[n]*cos(phi_ES) + v*vx[n]* sin(phi_ES) );
                vz0 = vz[n]*cos(theta_ES) - sqrt( v*v - vz[n]*vz[n] )*sin(theta_ES)*cos(phi_ES);
            }
            else{  
                vx0 = v*sin(theta_ES)*cos(phi_ES);
                vy0 = v*sin(theta_ES)*sin(phi_ES);
                vz0 = v*cos(theta_ES);
            }
            vx[n] = vx0;
            vy[n] = vy0;
            vz[n] = vz0;
        }
        else if(p2<b && energy[n]>I/1000){
            //ionization
            T = energy[n] * 1000; //unit: eV
            T_a = 1000;
            T_b = 2*I;
            t_b = I;
            T_s = 4.17;
            t_s = 13.8;
            T_0 = T_s - T_a / (T+T_b);
            t = t_s*T/(T+t_b);
            T_m = (T-I)/2;
            Rn = curand_uniform_double(&state);
            k_1 = atan((T_m-T_0)/t);
            k_2 = atan(T_0/t);
            k0 = T_0 + t * tan( Rn*(k_1+k_2) - k_2 );
            E_1 = k0;
            E_2 = T - E_1 - I;
            if(E_2>E_1){
                E_p = E_2;
                E_s = E_1;
            }
            else{
                E_p = E_1;
                E_s = E_2;
            }
            E_p = E_p * e;
            E_s = E_s * e;
            phi_p = 2*pi* curand_uniform_double(&state);
            phi_s = phi_p - pi;
            sin_theta_p = sqrt( (k0/T) / ((1-k0/T)*T/(2*E_e)+1) );
            theta_p = asin(sin_theta_p);
            sin_theta_s = sqrt( (1-k0/T)/(1+k0/(2*E_e)) );
            theta_s = asin(sin_theta_s);
            gamma = 1 + E_p/(m*c*c);
            v_p = c*sqrt(1 - 1/(gamma*gamma));
            ratio = v_p/v_e;
            vx[n] = ratio*vx[n];
            vy[n] = ratio*vy[n];
            vz[n] = ratio*vz[n];

            v = sqrt( vx[n]*vx[n] + vy[n]*vy[n] + vz[n]*vz[n] );
            if( v*v - vz[n]*vz[n] > 0.0001){
                vx0 = vx[n]*cos(theta_p) + sin(theta_p)/sqrt(v*v-vz[n]*vz[n])*( vx[n]*vz[n]*cos(phi_p) - v*vy[n]* sin(phi_p) );
                vy0 = vy[n]*cos(theta_p) + sin(theta_p)/sqrt(v*v-vz[n]*vz[n])*( vy[n]*vz[n]*cos(phi_p) + v*vx[n]* sin(phi_p) );
                vz0 = vz[n]*cos(theta_p) - sqrt( v*v - vz[n]*vz[n] )*sin(theta_p)*cos(phi_p);
            }
            else{  
                vx0 = v*sin(theta_p)*cos(phi_p);
                vy0 = v*sin(theta_p)*sin(phi_p);
                vz0 = v*cos(theta_p);
            }
            vx[n] = vx0;
            vy[n] = vy0;
            vz[n] = vz0;

            //secondary e
            gamma = 1 + E_s/(m*c*c);
            v_s = c*sqrt(1 - 1/(gamma*gamma));
            secondaryParticle[n] = 1;
            secondary_vx[n] = v_s*sin(theta_s)*cos(phi_s);
            secondary_vy[n] = v_s*sin(theta_s)*sin(phi_s);
            secondary_vz[n] = v_s*cos(theta_s);
            particleSamplingOut[n] = index;
            ion1_vx[n] = curand_normal(&state)/sqrt(m_ion/(kb*temperature[index]));
            ion1_vy[n] = curand_normal(&state)/sqrt(m_ion/(kb*temperature[index]));
            ion1_vz[n] = curand_normal(&state)/sqrt(m_ion/(kb*temperature[index]));
        }
        else{
            //excitation
            q0 = 6.514e-14;
            T = energy[n] * 1000; //unit: eV
            k_1 = 0;
            k_2 = 0;
            for(i=0; i<lenExc; i++){
                aExc = q0 * Aexc[i]/(kExc[i]*kExc[i]);
                bExc = powf( kExc[i]/T, omegaExc[i] );
                cExc = powf( powf( 1-kExc[i]/T, gammaExc[i] ), vExc[i]);
                k_1 = k_1 + aExc*bExc*cExc*kExc[i];
                k_2 = k_2 + aExc*bExc*cExc;
            }
            E_dep = (k_1/k_2); //unit: eV
            E_p = T - E_dep; //unit: eV
            gamma = 1 + E_p*e/(m*c*c);
            v_p = c*sqrt(1 - 1/(gamma*gamma));
            ratio = v_p/v_e;
            vx[n] = ratio*vx[n];
            vy[n] = ratio*vy[n];
            vz[n] = ratio*vz[n];
        }
    }
}
__global__ void processMandelbrotElement( 
                     double *x,
                     double *y,
                     double *z,
                     double *vx,
                     double *vy,
                     double *vz,
                     double *sig_elastic,
                     double *sig_ionization,
                     double *sig_excitation,
                     double *energy,
                     double *xInterp,
                     double *yInterp,
                     double *V_interp,
                     double *density,
                     double *temperature,
                     double *dphi,
                     double *kExc,
                     double *Aexc,
                     double *omegaExc,
                     double *vExc,
                     double *gammaExc,
                     bool *secondaryParticle,
                     double *secondary_vx,
                     double *secondary_vy,
                     double *secondary_vz,
                     double *particleSamplingOut,
                     double *ion1_vx,
                     double *ion1_vy,
                     double *ion1_vz,
                     int *sum_cell0,
                     const int lenExc,
                     const int cellnumxy,
                     const double dr,
                     const double dz,
                     const double dt,
                     const int parNum) {
        EM1(x,y,z,vx,vy,vz,sig_elastic,sig_ionization,sig_excitation,energy,
            xInterp,yInterp,V_interp,density,temperature,dphi,kExc,Aexc,omegaExc,vExc,
            gammaExc,secondaryParticle,secondary_vx,secondary_vy,secondary_vz,particleSamplingOut,ion1_vx,ion1_vy,ion1_vz,
            sum_cell0,lenExc,cellnumxy,dr,dz,dt,parNum);
}
