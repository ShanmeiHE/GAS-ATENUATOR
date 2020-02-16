//nvcc -ptx EM7.cu -ccbin "F:Visual Studio\VC\Tools\MSVC\14.12.25827\bin\Hostx64\x64"
#include "sm_20_atomic_functions.h"

#ifndef CAFFE_COMMON_CUH_
#define CAFFE_COMMON_CUH_

#include <cuda.h>
 
    #if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600

    #else
    static __inline__ __device__ double atomicAdd(double *address, double val) {
        unsigned long long int* address_as_ull = (unsigned long long int*)address;
        unsigned long long int old = *address_as_ull, assumed;
        if (val==0.0)
            return __longlong_as_double(old);
        do {
            assumed = old;
            old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val +__longlong_as_double(assumed)));
        } while (assumed != old);
        return __longlong_as_double(old);
	}



    #endif
#endif

__device__ void EM1( double *r,
                     double *z,
                     double * charge,
                     double * vr,
                     double * vz,
                     double * jr,
                     double * jz,
                     int * parDelete,
                     const int parNum,
                     const int gridR,
                     const int gridZ,
                     const double dr,
                     const double dz,
                     const double FN,
                     const double PHIp) {
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
            a1 = 0;
            a2 = 0;
            a3 = 0;
            a4 = (ar+1) + (az+1) * gridR;

            b1 = 0;
            b2 = 0;
            b3 = 0;
            b4 = 1;
        }
        else if (az>=gridZ-1){
            a1 = 0;
            a2 = (ar+1) + az * gridR;
            a3 = 0;
            a4 = 0;

            b1 = 0;
            b2 = 1;
            b3 = 0;
            b4 = 0;
        }
        else{
            a1 = 0;
            a2 = (ar+1) + az * gridR;
            a3 = 0;
            a4 = (ar+1) + (az+1) * gridR;
        
            b1 = 0;
            b2 = ((az+1.5)*dz-z0)/(dz);
            b3 = 0;
            b4 = (z0-(az+0.5)*dz)/(dz);  
        }
    }
    else if (ar>=gridR-1){
        if( az<0 ){
            a1 = 0;
            a2 = 0;
            a3 = ar + (az+1) * gridR;
            a4 = 0;

            b1 = 0;
            b2 = 0;
            b3 = 1;
            b4 = 0;
        }
        else if (az>=gridZ-1){
            a1 = ar + az * gridR;
            a2 = 0;
            a3 = 0;
            a4 = 0;

            b1 = 1;
            b2 = 0;
            b3 = 0;
            b4 = 0;
        }
        else{
            a1 = ar + az * gridR;
            a2 = 0;
            a3 = ar + (az+1) * gridR;
            a4 = 0;
    
            b1 = ((az+1.5)*dz-z0)/(dz);
            b2 = 0;
            b3 = (z0-(az+0.5)*dz)/(dz);
            b4 = 0;    
        }
    }
    else{
        if( az<0 ){
            a1 = 0;
            a2 = 0;
            a3 = ar + (az+1) * gridR;
            a4 = (ar+1) + (az+1) * gridR;

            b1 = 0;
            b2 = 0;
            b3 = ((ar+1.5)*dr-r0)/(dr);
            b4 = (r0-(ar+0.5)*dr)/(dr);   
        }
        else if (az>=gridZ-1){
            a1 = ar + az * gridR;
            a2 = (ar+1) + az * gridR;
            a3 = 0;
            a4 = 0;

            b1 = ((ar+1.5)*dr-r0)/(dr);
            b2 = (r0-(ar+0.5)*dr)/(dr);  
            b3 = 0;
            b4 = 0;
        }
        else{
            a1 = ar + az * gridR;
            a2 = (ar+1) + az * gridR;
            a3 = ar + (az+1) * gridR;
            a4 = (ar+1) + (az+1) * gridR;
    
            b1 = ((ar+1.5)*dr-r0)*((az+1.5)*dz-z0)/(dr*dz);
            b2 = (r0-(ar+0.5)*dr)*((az+1.5)*dz-z0)/(dr*dz);
            b3 = ((ar+1.5)*dr-r0)*(z0-(az+0.5)*dz)/(dr*dz);
            b4 = (r0-(ar+0.5)*dr)*(z0-(az+0.5)*dz)/(dr*dz);    
        }
    }
    if (parDelete[n]==1){
        a1 = 0;
        a2 = 0;
        a3 = 0;
        a4 = 0;

        b1 = 0;
        b2 = 0;
        b3 = 0;
        b4 = 0;
    }
    double V1,V2;
    V1 = PHIp/2*( ( ((ar+1)*dr)*((ar+1)*dr) - (ar*dr)*(ar*dr) )*dz );
    V2 = PHIp/2*( ( ((ar+2)*dr)*((ar+2)*dr) - ((ar+1)*dr)*((ar+1)*dr) )*dz );
    
    double *J;
    J = jr;
    J = J+a1;
    atomicAdd(J, FN*b1*charge[n]*vr[n]/V1);
    J = jr;
    J = J+a2;
    atomicAdd(J, FN*b2*charge[n]*vr[n]/V2);
    J = jr;
    J = J+a3;
    atomicAdd(J, FN*b3*charge[n]*vr[n]/V1);
    J = jr;
    J = J+a4;
    atomicAdd(J, FN*b4*charge[n]*vr[n]/V2);
    J = jz;
    J = J+a1;
    atomicAdd(J, FN*b1*charge[n]*vz[n]/V1);
    J = jz;
    J = J+a2;
    atomicAdd(J, FN*b2*charge[n]*vz[n]/V2);
    J = jz;
    J = J+a3;
    atomicAdd(J, FN*b3*charge[n]*vz[n]/V1);
    J = jz;
    J = J+a4;
    atomicAdd(J, FN*b4*charge[n]*vz[n]/V2);
    
}


__global__ void processMandelbrotElement( 
                     double *r,
                     double *z,
                     double * charge,
                     double * vr,
                     double * vz,
                     double * jr,
                     double * jz,
                     int * parDelete,
                     const int parNum,
                     const int gridR,
                     const int gridZ,
                     const double dr,
                     const double dz,
                     const double FN,
                     const double PHIp) {
        EM1(r, z, charge, vr, vz, jr, jz, parDelete, parNum, gridR, gridZ, dr, dz, FN, PHIp);
}
