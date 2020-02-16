//nvcc -ptx test.cu -ccbin "F:Visual Studio\VC\Tools\MSVC\14.12.25827\bin\Hostx64\x64"
#include "curand_kernel.h"
__device__ void EM1( double *x,
                     double *y,
                     const int parNum) {
                      int globalBlockIndex = blockIdx.x + blockIdx.y * gridDim.x;
    int localThreadIdx = threadIdx.x + blockDim.x * threadIdx.y;
    int threadsPerBlock = blockDim.x*blockDim.y;
    int n = localThreadIdx + globalBlockIndex*threadsPerBlock;
    if ( n >= parNum ){
        return;
    }
    curandState state;
    curand_init((unsigned long long)clock(),0,n, & state);
    x[n]=curand_uniform_double(&state); 
    y[n]=curand_normal(&state); 
}

__global__ void processMandelbrotElement( 
                     double *x,
                     double *y,
                     const int parNum) {
        EM1(x,y,parNum);
}