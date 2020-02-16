//nvcc -ptx EM4.cu -ccbin "F:Visual Studio\VC\Tools\MSVC\14.12.25827\bin\Hostx64\x64"
__device__ void EM1( double * x, 
                     double * y, 
                     double * z, 
                     double * vx,
                     double * vy,
                     double * vz,
                     double * r,
                     double * phi,
                     double * vr,
                     double * Er,
                     double * Ez,
                     double * Hphi,
                     double * charge,
                     double * m,
                     double * E,
                     int *parDelete,
                     const int parNum,
                     const double Rp,
                     const double Lp,
                     const double PHIp,
                     const int gridR,
                     const int gridZ,
                     const double mu,
                     const double c,
                     const double dr,
                     const double dz,
                     const double dt ) {
    int globalBlockIndex = blockIdx.x + blockIdx.y * gridDim.x;
    int localThreadIdx = threadIdx.x + blockDim.x * threadIdx.y;
    int threadsPerBlock = blockDim.x*blockDim.y;
    int n = localThreadIdx + globalBlockIndex*threadsPerBlock;
    if ( n >= parNum || parDelete[n]==1){
        return;
    }
    double r0 = r[n];
    double z0 = z[n];
    int ar,az,a1,a2,a3,a4;
    double Er0,Ez0,Hphi0;

    // Er
    ar = floor(r0/dr-0.5);
    az = floor(z0/dz);
    a1 = ar + az * gridR;
    a2 = (ar+1) + az * gridR;
    a3 = ar + (az+1) * gridR;
    a4 = (ar+1) + (az+1) * gridR;
    if( ar<0 ){
        Er0 = ( Er[a2]*(az*dz+dz-z0) + Er[a4]*(z0-az*dz) )/dz;
    }
    else if( ar >= gridR-1 ){
        Er0 = ( Er[a1]*(az*dz+dz-z0) + Er[a3]*(z0-az*dz) )/dz;
    }
    else{
        Er0 = ( Er[a1]*((ar+1.5)*dr-r0)*((az+1)*dz-z0)
                 + Er[a2]*(r0-(ar+0.5)*dr)*((az+1)*dz-z0)
                 + Er[a3]*((ar+1.5)*dr-r0)*(z0-az*dz)
                 + Er[a4]*(r0-(ar+0.5)*dr)*(z0-az*dz) )/(dr*dz);
        
    }
    // Ez
    ar = floor(r0/dr);
    az = floor(z0/dz-0.5);
    a1 = ar + az * (gridR+1);
    a2 = (ar+1) + az * (gridR+1);
    a3 = ar + (az+1) * (gridR+1);
    a4 = (ar+1) + (az+1) * (gridR+1);
    if( az<0 ){
        Ez0 = ( Ez[a3]*(ar*dr+dr-r0) + Ez[a4]*(r0-ar*dr) )/dr;
    }
    else if( az >= gridZ-1 ){
        Ez0 = ( Ez[a1]*(ar*dr+dr-r0) + Ez[a2]*(r0-ar*dr) )/dr;
    }
    else{
        Ez0 = ( Ez[a1]*((ar+1)*dr-r0)*((az+1.5)*dz-z0)
                 + Ez[a2]*(r0-ar*dr)*((az+1.5)*dz-z0)
                 + Ez[a3]*((ar+1)*dr-r0)*(z0-(az+0.5)*dz)
                 + Ez[a4]*(r0-ar*dr)*(z0-(az+0.5)*dz) )/(dr*dz);
        
    }

    // Hphi
    ar = floor(r0/dr-0.5);
    az = floor(z0/dz-0.5);
    a1 = ar + az * gridR;
    a2 = (ar+1) + az * gridR;
    a3 = ar + (az+1) * gridR;
    a4 = (ar+1) + (az+1) * gridR;
    if( ar<0 ){
        if( az<0 ){
            Hphi0 = Hphi[a4];
        }
        else if( az>=gridZ-1 ){
            Hphi0 = Hphi[a2];
        }
        else{
            Hphi0 = ( Hphi[a2]*((az+1.5)*dz-z0) + Hphi[a4]*(z0-(az+0.5)*dz) )/dz;
        }
    }
    else if( ar>=gridR-1 ){
        if( az<0 ){
            Hphi0 = Hphi[a3];
        }
        else if( az>=gridZ-1 ){
            Hphi0 = Hphi[a1];
        }
        else{
            Hphi0 = ( Hphi[a1]*((az+1.5)*dz-z0) + Hphi[a3]*(z0-(az+0.5)*dz) )/dz;
        }
    }
    else if( az<0 ){
        Hphi0 = ( Hphi[a3]*((ar+1.5)*dr-r0) + Hphi[a4]*(r0-(ar+0.5)*dr) )/dr;
    }
    else if( az>=gridZ-1 ){
        Hphi0 = ( Hphi[a1]*((ar+1.5)*dr-r0) + Hphi[a2]*(r0-(ar+0.5)*dr) )/dr;
    }
    else{
        Hphi0 = ( Hphi[a1]*((ar+1.5)*dr-r0)*((az+1.5)*dz-z0)
               + Hphi[a2]*(r0-(ar+0.5)*dr)*((az+1.5)*dz-z0)
               + Hphi[a3]*((ar+1.5)*dr-r0)*(z0-(az+0.5)*dz)
               + Hphi[a4]*(r0-(ar+0.5)*dr)*(z0-(az+0.5)*dz) )/(dr*dz);
    }

    //F
    double Fx,Fy,Fz,Fr;
    Fr = charge[n] * (Er0 + (-vz[n]*Hphi0*mu));
    Fz = charge[n] * (Ez0 + (vr[n]*Hphi0*mu));
    Fx = Fr*cos(phi[n]);
    Fy = Fr*sin(phi[n]);


    //v
    double gamma,ux,uy,uz;
    gamma = 1/sqrt( 1-( vx[n]*vx[n] + vy[n]*vy[n] + vz[n]*vz[n] )/(c*c) );
    ux = gamma * vx[n] + Fx/m[n]*dt;
    uy = gamma * vy[n] + Fy/m[n]*dt;
    uz = gamma * vz[n] + Fz/m[n]*dt;
    gamma = sqrt( 1+ (ux*ux + uy*uy + uz*uz)/(c*c) );

    E[n] = (gamma-1)*m[n]*c*c;

    vx[n] = ux/gamma;
    vy[n] = uy/gamma;
    vz[n] = uz/gamma;

    x[n] = x[n] + 0.5*dt*vx[n];
    y[n] = y[n] + 0.5*dt*vy[n];
    z[n] = z[n] + 0.5*dt*vz[n];

    r[n] = sqrt( x[n]*x[n] + y[n]*y[n] );
    phi[n] = atan( y[n]/x[n] );
    if (r[n] > Rp){
        parDelete[n] = 1;
    }
    double vx1;
    while(phi[n]<0){
        phi[n] = phi[n] + PHIp;
        vx1 = vx[n] * cos(PHIp) - vy[n] * sin(PHIp);
        vy[n] = vx[n] * sin(PHIp) + vy[n] * cos(PHIp);
        vx[n] = vx1;
    }
    while(phi[n]>PHIp){
        phi[n] = phi[n] - PHIp;
        vx1 = vx[n] * cos(PHIp) + vy[n] * sin(PHIp);
        vy[n] = -vx[n] * sin(PHIp) + vy[n] * cos(PHIp);
        vx[n] = vx1;
    }
    x[n] = r[n] * cos(phi[n]);
    y[n] = r[n] * sin(phi[n]);
    if (z[n]>Lp){
        z[n] = 2*Lp - z[n];
        vz[n] = -vz[n];
    }
    if (z[n]<0){
        z[n] = -z[n];
        vz[n] = -vz[n];
    }
}


__global__ void processMandelbrotElement( 
                     double * x, 
                     double * y, 
                     double * z, 
                     double * vx,
                     double * vy,
                     double * vz,
                     double * r,
                     double * phi,
                     double * vr,
                     double * Er,
                     double * Ez,
                     double * Hphi,
                     double * charge,
                     double * m,
                     double * E,
                     int *parDelete,
                     const int parNum,
                     const double Rp,
                     const double Lp,
                     const double PHIp,
                     const int gridR,
                     const int gridZ,
                     const double mu,
                     const double c,
                     const double dr,
                     const double dz,
                     const double dt ) {
        EM1(x, y, z, vx, vy, vz, r, phi, vr, Er, Ez, Hphi, charge, m, E, parDelete, parNum, Rp, Lp, PHIp, gridR, gridZ, mu, c, dr, dz, dt);
}
