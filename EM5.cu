//nvcc -ptx EM5.cu -ccbin "F:Visual Studio\VC\Tools\MSVC\14.12.25827\bin\Hostx64\x64"
__device__ void EM1( double *r,
                     double *z,
                     double * ar0,
                     double * br0,
                     double * az0,
                     double * bz0,
                     const int parNum,
                     const int gridR,
                     const int gridZ,
                     const double dr,
                     const double dz ) {
    int globalBlockIndex = blockIdx.x + blockIdx.y * gridDim.x;
    int localThreadIdx = threadIdx.x + blockDim.x * threadIdx.y;
    int threadsPerBlock = blockDim.x*blockDim.y;
    int n = localThreadIdx + globalBlockIndex*threadsPerBlock;
    if ( n >= parNum ){
        return;
    }
    double r0 = r[n];
    double z0 = z[n];
    int ar,az,a1,a2,a3,a4;
    double b1,b2,b3,b4;
    // Er
    ar = floor(r0/dr-0.5);
    az = floor(z0/dz);
    a1 = ar + az * gridR + 1;
    a2 = (ar+1) + az * gridR + 1;
    a3 = ar + (az+1) * gridR + 1;
    a4 = (ar+1) + (az+1) * gridR + 1;
    if( ar<0 ){
        b1 = 0;
        b2 = (az*dz+dz-z0)/dz;
        b3 = 0;
        b4 = (z0-az*dz)/dz;
        a1 = 1;
        a3 = 1;
    }
    else if( ar >= gridR-1 ){
        b1 = (az*dz+dz-z0)/dz;
        b2 = 0;
        b3 = (z0-az*dz)/dz;
        b4 = 0;
        a2 = 1;
        a4 = 1;
    }
    else{
        b1 = ((ar+1.5)*dr-r0)*((az+1)*dz-z0)/(dr*dz);
        b2 = (r0-(ar+0.5)*dr)*((az+1)*dz-z0)/(dr*dz);
        b3 = ((ar+1.5)*dr-r0)*(z0-az*dz)/(dr*dz);
        b4 = (r0-(ar+0.5)*dr)*(z0-az*dz)/(dr*dz);    
    }
    ar0[n] = a1;
    ar0[n+parNum] = a2;
    ar0[n+2*parNum] = a3;
    ar0[n+3*parNum] = a4;

    br0[n] = b1;
    br0[n+parNum] = b2;
    br0[n+2*parNum] = b3;
    br0[n+3*parNum] = b4;
    // Ez
    ar = floor(r0/dr);
    az = floor(z0/dz-0.5);
    a1 = ar + az * (gridR+1) + 1;
    a2 = (ar+1) + az * (gridR+1) + 1;
    a3 = ar + (az+1) * (gridR+1) + 1;
    a4 = (ar+1) + (az+1) * (gridR+1) + 1;
    if( az<0 ){
        b1 = 0;
        b2 = 0;
        b3 = (ar*dr+dr-r0)/dr;
        b4 = (r0-ar*dr)/dr;
        a1 = 1;
        a2 = 1;
    }
    else if( az >= gridZ-1 ){
        b1 = (ar*dr+dr-r0)/dr;
        b2 = (r0-ar*dr)/dr;
        b3 = 0;
        b4 = 0;
        a3 = 1;
        a4 = 1;
    }
    else{
        b1 = ((ar+1)*dr-r0)*((az+1.5)*dz-z0)/(dr*dz);
        b2 = (r0-ar*dr)*((az+1.5)*dz-z0)/(dr*dz);
        b3 = ((ar+1)*dr-r0)*(z0-(az+0.5)*dz)/(dr*dz);
        b4 = (r0-ar*dr)*(z0-(az+0.5)*dz)/(dr*dz);
        
    }
    az0[n] = a1;
    az0[n+parNum] = a2;
    az0[n+2*parNum] = a3;
    az0[n+3*parNum] = a4;

    bz0[n] = b1;
    bz0[n+parNum] = b2;
    bz0[n+2*parNum] = b3;
    bz0[n+3*parNum] = b4;
}


__global__ void processMandelbrotElement( 
                     double *r,
                     double *z,
                     double * ar0,
                     double * br0,
                     double * az0,
                     double * bz0,
                     const int parNum,
                     const int gridR,
                     const int gridZ,
                     const double dr,
                     const double dz ) {
        EM1(r, z, ar0, br0, az0, bz0, parNum, gridR, gridZ, dr, dz);
}
