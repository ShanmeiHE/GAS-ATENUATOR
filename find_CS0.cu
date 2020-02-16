//nvcc -ptx find_CS0.cu -ccbin "F:Visual Studio\VC\Tools\MSVC\14.12.25827\bin\Hostx64\x64"
__device__ void EM1( double * E_PE,
                     double * x_el,
                     double * y_el,
                     double * x_sel,
                     double * y_sel,
                     double * x_ion,
                     double * y_ion,
                     double * k,
                     double * A_2,
                     double * omega_2,
                     double * v_2,
                     double * gamma_2,
                     double * sigma_el,
                     double * sigma_ion,
                     double * sigma_exc,
                     double sqr_A0,
                     double sqr_a0,
                     double q0,
                     int len,
                     int parNum) {
    int globalBlockIndex = blockIdx.x + blockIdx.y * gridDim.x;
    int localThreadIdx = threadIdx.x + blockDim.x * threadIdx.y;
    int threadsPerBlock = blockDim.x*blockDim.y;
    int n = localThreadIdx + globalBlockIndex*threadsPerBlock;
    if ( n >= parNum ){
        return;
    }

    //Elastic
    double e = E_PE[n];
    double a,b,sigma;
    int i;
    
    if ( e >= 0.05 ){
        i = 0;
        while( e>x_el[i] && e<x_el[i+1] ){
            i++;
        }
        a = ( x_el[i+1]-e )/( x_el[i+1] - x_el[i] );
        b = ( e-x_el[i] )/( x_el[i+1] - x_el[i] );
        sigma = (a*y_el[i] + b*y_el[i]) * sqr_a0 * 2;
    }
    else{
        
        e = 1000*e;
        i = 0;
        while( e>x_sel[i] && e<x_sel[i+1] ){
            i++;
        }
        a = ( x_sel[i+1]-e )/( x_sel[i+1] - x_sel[i] );
        b = ( e-x_sel[i] )/( x_sel[i+1] - x_sel[i] );
        sigma = (a*y_sel[i] + b*y_sel[i]) * sqr_A0;
    }
    sigma_el[n] = sigma;
    
    //Electron impact ionization
    e = 1000 * E_PE[n];
    i = 0;
    while( e>x_ion[i] && e<x_ion[i+1] ){
       i++;
    }
    a = ( x_ion[i+1]-e )/( x_ion[i+1] - x_ion[i] );
    b = ( e-x_ion[i] )/( x_ion[i+1] - x_ion[i] );
    sigma_ion[n] = (a*y_ion[i] + b*y_ion[i]) * sqr_A0;

    //Electron impact excitation
    e = E_PE[n];
    sigma = 0;
    for (i = 0;i<len ;i++ ){
        sigma = sigma + (q0 * A_2[i]/powf(k[i],2))*  powf(k[i]/e,omega_2[i]) * powf(powf(1-k[i]/e,gamma_2[i]),v_2[i]);
    }
    sigma_exc[n] = sigma/10000;
}
__global__ void processMandelbrotElement( 
                     double * E_PE,
                     double * x_el,
                     double * y_el,
                     double * x_sel,
                     double * y_sel,
                     double * x_ion,
                     double * y_ion,
                     double * k,
                     double * A_2,
                     double * omega_2,
                     double * v_2,
                     double * gamma_2,
                     double * sigma_el,
                     double * sigma_ion,
                     double * sigma_exc,
                     double sqr_A0,
                     double sqr_a0,
                     double q0,
                     int len,
                     int parNum) {
        EM1(E_PE,x_el,y_el,x_sel,y_sel,x_ion,y_ion,k,A_2,omega_2,v_2,gamma_2,sigma_el,sigma_ion,sigma_exc,sqr_A0,sqr_a0,q0,len,parNum);
}
