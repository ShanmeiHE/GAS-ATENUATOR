//nvcc -ptx EM3_2.cu -ccbin "F:Visual Studio\VC\Tools\MSVC\14.12.25827\bin\Hostx64\x64"
__device__ void EM1( double * Er,
                     double * Ez,
                     double * Hphi,
                     const double mu,
                     const double dr,
                     const double dz,
                     const double dt ) {
    int nz = blockIdx.x + blockIdx.y * gridDim.x;
    int nr = threadIdx.x + blockDim.x * threadIdx.y;
    int threadsPerBlock = blockDim.x*blockDim.y;
    int n_r = nr + nz*threadsPerBlock;  
    int n_z = nr + nz*(threadsPerBlock+1);

    Hphi[n_r] = Hphi[n_r] -0.5* (dt/(mu*dr)*( Ez[n_z+1]-Ez[n_z] ) - dt/(mu*dz)*( Er[n_r + threadsPerBlock]-Er[n_r] ));
    
}
__global__ void processMandelbrotElement( 
                      double * Er,
                     double * Ez,
                     double * Hphi,
                     const double mu,
                     const double dr,
                     const double dz,
                     const double dt ) {
    EM1(Er,Ez,Hphi,mu,dr,dz,dt);
}
