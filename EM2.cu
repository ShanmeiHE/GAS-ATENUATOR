//nvcc -ptx EM2.cu -ccbin "F:Visual Studio\VC\Tools\MSVC\14.12.25827\bin\Hostx64\x64"
__device__ void EM1( double * x, 
                     double * y, 
                     double * z, 
                     double * vx,
                     double * vy,
                     double * vz,
                     double * r,
                     double * phi,
                     double * vr,
                     int * parDelete,
                     const int parNum,
                     const double Rp,
                     const double Lp,
                     const double PHIp,
                     const double dr,
                     const double dz,
                     const double dt ) {
    int globalBlockIndex = blockIdx.x + blockIdx.y * gridDim.x;
    int localThreadIdx = threadIdx.x + blockDim.x * threadIdx.y;
    int threadsPerBlock = blockDim.x*blockDim.y;
    int n = localThreadIdx + globalBlockIndex*threadsPerBlock;
    if ( n >= parNum || parDelete[n]==1 ){
        return;
    }
    x[n] =  x[n] + 0.5 * vx[n] * dt;
    y[n] =  y[n] + 0.5 * vy[n] * dt;
    z[n] =  z[n] + 0.5 * vz[n] * dt;
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
    vr[n] = vx[n]*cos(phi[n]) + vy[n]*sin(phi[n]) ;
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
                      int * parDelete,
                      const int parNum,
                      const double Rp,
                      const double Lp,
                      const double PHIp,
                      const double dr,
                      const double dz,
                      const double dt ) {
        EM1(x, y, z, vx, vy, vz, r, phi, vr, parDelete, parNum, Rp, Lp, PHIp, dr, dz, dt);
}
