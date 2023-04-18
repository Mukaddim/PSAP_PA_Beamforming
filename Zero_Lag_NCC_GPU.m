function NCC_gpu = Zero_Lag_NCC_GPU(data1,data2,kernelY,kernelX,upsample_factor)

    [samples,lines] = size(data1);
    [X,Y]=meshgrid(1:lines,1:samples);
    [Xq,Yq]=meshgrid(linspace(1,lines,lines*upsample_factor(2)),linspace(1,samples,samples*upsample_factor(1)));
    data1 = interp2(X,Y,data1,Xq,Yq);
    data2 = interp2(X,Y,data2,Xq,Yq);
    
    halfY = (kernelY-1)/2;
    halfX = (kernelX-1)/2;
    
    data1_pad = padarray(data1,[halfY,halfX]);
    data2_pad = padarray(data2,[halfY,halfX]);
    
    [numY,numX] = size(data1);
    [samples,lines] = size(data1_pad);
    
    data1_g = gpuArray(data1_pad);
    data2_g = gpuArray(data2_pad);
    NCC_gpu = gpuArray.zeros(numY, numX);
    
    
    kernel = parallel.gpu.CUDAKernel('NCC_zero_lag.ptx','NCC_zero_lag.cu','Zero_Lag_NCC_GPU');
    kernel.ThreadBlockSize = [32 32 1];
    kernel.GridSize = [round(numX/32)+1 round(numY/32)+1 1];
    
    NCC_gpu = feval(kernel,NCC_gpu,data1_g,data2_g,halfX,halfY,kernelX,kernelY,samples,lines,numX,numY);
    
    
  
    
    
end