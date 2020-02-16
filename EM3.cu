//nvcc -ptx EM3.cu -ccbin "F:Visual Studio\VC\Tools\MSVC\14.12.25827\bin\Hostx64\x64"
__device__ void EM1( double * Er0,
                     double * Ez0,
                     double * Hphi0,
                     double * Er,
                     double * Ez,
                     double * jr,
                     double * jz,
                     const double mu,
                     const double epsilon,
                     const double dr,
                     const double dz,
                     const double dt ) {
    int nz = blockIdx.x + blockIdx.y * gridDim.x;
    int nr = threadIdx.x + blockDim.x * threadIdx.y;
    int threadsPerBlock = blockDim.x*blockDim.y;
    int BlocksPerGrid = gridDim.x*gridDim.y;
    int n_r = nr + nz*(threadsPerBlock-1);  
    int n_z = nr + nz*threadsPerBlock;
    if( nr < threadsPerBlock-1 ){
        if( nz < 1){
            Er[n_r] = Er0[n_r + (threadsPerBlock-1)];
        }
        else{
            if( nz >= BlocksPerGrid-1 ){
                Er[n_r] = Er0[n_r - (threadsPerBlock-1)];
            }
            else{
                Er[n_r] = Er0[n_r] - dt/epsilon*(jr[n_r-(threadsPerBlock-1)]+jr[n_r])/2 - dt/(epsilon*dz)*( Hphi0[n_r]-Hphi0[n_r - (threadsPerBlock-1)] );
            }
        }
    }
    if( nz < BlocksPerGrid-1 ){
        if( nr < 1 ){
            Ez[n_z] = Ez0[n_z] + dt/epsilon*jz[n_r] + 4*dt/(epsilon*dr)*Hphi0[n_r];
        }
        else{
            if( nr < threadsPerBlock-1 ){
                Ez[n_z] = Ez0[n_z] - dt/epsilon*(jz[n_r-1]+jz[n_r])/2 + dt/epsilon*(1/(2*(nr+1)*dr)+1/dr)*Hphi0[n_r] + dt/epsilon*(1/(2*(nr+1)*dr)-1/dr)*Hphi0[n_r-1] ;
            }
        }
    }
}
__global__ void processMandelbrotElement( 
                      double * Er0,
                      double * Ez0,
                      double * Hphi0,
                      double * Er,
                      double * Ez,
                      double * jr,
                      double * jz,
                      const double mu,
                      const double epsilon,
                      const double dr,
                      const double dz,
                      const double dt ) {
    EM1(Er0,Ez0,Hphi0,Er,Ez,jr,jz,mu,epsilon,dr,dz,dt);
}
