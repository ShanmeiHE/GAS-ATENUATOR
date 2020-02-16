//nvcc -ptx EM6.cu -ccbin "F:Visual Studio\VC\Tools\MSVC\14.12.25827\bin\Hostx64\x64"
__device__ void EM1( double *r,
                     double *z,
                     double * a,
                     double * b,
                     int * parDelete,
                     const int parNum,
                     const int gridR,
                     const int gridZ,
                     const double dr,
                     const double dz) {
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
    
    ar = floor(r0/dr-0.5);
    az = floor(z0/dz-0.5);
    if (ar<0){
        if (az<0){
            a1 = 1;
            a2 = 1;
            a3 = 1;
            a4 = (ar+1) + (az+1) * gridR + 1;

            b1 = 0;
            b2 = 0;
            b3 = 0;
            b4 = 1;
        }
        else if (az>=gridZ-1){
            a1 = 1;
            a2 = (ar+1) + az * gridR + 1;
            a3 = 1;
            a4 = 1;

            b1 = 0;
            b2 = 1;
            b3 = 0;
            b4 = 0;
        }
        else{
            a1 = 1;
            a2 = (ar+1) + az * gridR + 1;
            a3 = 1;
            a4 = (ar+1) + (az+1) * gridR + 1;
        
            b1 = 0;
            b2 = ((az+1.5)*dz-z0)/(dz);
            b3 = 0;
            b4 = (z0-(az+0.5)*dz)/(dz);  
        }
    }
    else if (ar>=gridR-1){
        if( az<0 ){
            a1 = 1;
            a2 = 1;
            a3 = ar + (az+1) * gridR + 1;
            a4 = 1;

            b1 = 0;
            b2 = 0;
            b3 = 1;
            b4 = 0;
        }
        else if (az>=gridZ-1){
            a1 = ar + az * gridR + 1;
            a2 = 1;
            a3 = 1;
            a4 = 1;

            b1 = 1;
            b2 = 0;
            b3 = 0;
            b4 = 0;
        }
        else{
            a1 = ar + az * gridR + 1;
            a2 = 1;
            a3 = ar + (az+1) * gridR + 1;
            a4 = 1;
    
            b1 = ((az+1.5)*dz-z0)/(dz);
            b2 = 0;
            b3 = (z0-(az+0.5)*dz)/(dz);
            b4 = 0;    
        }
    }
    else{
        if( az<0 ){
            a1 = 1;
            a2 = 1;
            a3 = ar + (az+1) * gridR + 1;
            a4 = (ar+1) + (az+1) * gridR + 1;

            b1 = 0;
            b2 = 0;
            b3 = ((ar+1.5)*dr-r0)/(dr);
            b4 = (r0-(ar+0.5)*dr)/(dr);   
        }
        else if (az>=gridZ-1){
            a1 = ar + az * gridR + 1;
            a2 = (ar+1) + az * gridR + 1;
            a3 = 1;
            a4 = 1;

            b1 = ((ar+1.5)*dr-r0)/(dr);
            b2 = (r0-(ar+0.5)*dr)/(dr);  
            b3 = 0;
            b4 = 0;
        }
        else{
            a1 = ar + az * gridR + 1;
            a2 = (ar+1) + az * gridR + 1;
            a3 = ar + (az+1) * gridR + 1;
            a4 = (ar+1) + (az+1) * gridR + 1;
    
            b1 = ((ar+1.5)*dr-r0)*((az+1.5)*dz-z0)/(dr*dz);
            b2 = (r0-(ar+0.5)*dr)*((az+1.5)*dz-z0)/(dr*dz);
            b3 = ((ar+1.5)*dr-r0)*(z0-(az+0.5)*dz)/(dr*dz);
            b4 = (r0-(ar+0.5)*dr)*(z0-(az+0.5)*dz)/(dr*dz);    
        }
    }
    if (parDelete[n]==1){
        a1 = 1;
        a2 = 1;
        a3 = 1;
        a4 = 1;

        b1 = 0;
        b2 = 0;
        b3 = 0;
        b4 = 0;
    }

    a[n] = a1;
    a[n+parNum] = a2;
    a[n+2*parNum] = a3;
    a[n+3*parNum] = a4;

    b[n] = b1;
    b[n+parNum] = b2;
    b[n+2*parNum] = b3;
    b[n+3*parNum] = b4;
    
}


__global__ void processMandelbrotElement( 
                     double *r,
                     double *z,
                     double * a,
                     double * b,
                     int * parDelete,
                     const int parNum,
                     const int gridR,
                     const int gridZ,
                     const double dr,
                     const double dz) {
        EM1(r, z, a, b, parDelete, parNum, gridR, gridZ, dr, dz);
}
