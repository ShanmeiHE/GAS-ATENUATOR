
    cudaFilename6 = 'EM7.cu';
    ptxFilename6 = 'EM7.ptx';
    kernel6 = parallel.gpu.CUDAKernel( ptxFilename6, cudaFilename6 );
    kernel6.ThreadBlockSize = [1024,1,1];
    kernel6.GridSize = [600,1,1];
    
    tic
    jr = zeros(H_phi_lenr, H_phi_lenz);
    jz = zeros(H_phi_lenr, H_phi_lenz);
    [~,~,~,~,~,jr,jz,~] = feval(kernel6, r, z, particle_charge, vr, vz, jr, jz, parDelete, particle_number, 100, 500, delta_r, delta_z, FN, PHIp);
    toc
    
    
    
