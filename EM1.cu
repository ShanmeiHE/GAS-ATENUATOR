/**
 * @file pctdemo_processMandelbrotElement.cu
 * 
 * CUDA code to calculate the Mandelbrot Set on a GPU.
 * 
 * Copyright 2011 The MathWorks, Inc.
 */

/** Work out which piece of the global array this thread should operate on */ 
__device__ size_t calculateGlobalIndex() {
    // Which block are we?
    size_t const globalBlockIndex = blockIdx.x + blockIdx.y * gridDim.x;
    // Which thread are we within the block?
    size_t const localThreadIdx = threadIdx.x + blockDim.x * threadIdx.y;
    // How big is each block?
    size_t const threadsPerBlock = blockDim.x*blockDim.y;
    // Which thread are we overall?
    return localThreadIdx + globalBlockIndex*threadsPerBlock;

}

/** The actual Mandelbrot algorithm for a single location */ 
__device__ double position( double const x0, 
                            double const vx0,
                            double const dt ) {
    int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x ;
    // Initialise: z = z0
    double x = x0;
    double vx = vx0;
    x =  x + 0.5 * vx * dt;
    return x;
}


/** Main entry point.
 * Works out where the current thread should read/write to global memory
 * and calls doIterations to do the actual work.
 */
__global__ void processMandelbrotElement( 
                      double * xi, 
                      double * yi,
                      double * zi,
                      double * vxi,
                      double * vyi,
                      double * vzi,
                      const double dt ) {
    // Work out which thread we are
    size_t const globalThreadIdx = calculateGlobalIndex();

    
    // Get our X and Y coords
    double const x = xi[globalThreadIdx];
    double const y = yi[globalThreadIdx];
    double const z = zi[globalThreadIdx];
    double const vx = vxi[globalThreadIdx];
    double const vy = vyi[globalThreadIdx];
    double const vz = vzi[globalThreadIdx];

    // Run the itearations on this location
    xi[globalThreadIdx] = position( x, vx, dt );
    yi[globalThreadIdx] = position( y, vy, dt );
    zi[globalThreadIdx] = position( z, vz, dt );
}

