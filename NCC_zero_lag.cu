__global__ void Zero_Lag_NCC_GPU (double* NCC,double* RF1,double* RF2,double halfX,double halfY,double kerX,double kerY,double samples,double alines,double numX,double numY)
{
	// NCC calculation at zero lag for dual aperture imaging
	int idx = threadIdx.x + blockDim.x*blockIdx.x;
	int idy = threadIdx.y + blockDim.y*blockIdx.y;
	
	if(idx<numX && idy<numY)
	{
		int start_idx = idx + (int)halfX;
		int start_idy = idy + (int)halfY;
		
		double RF_mean1 = 0.0;
		double RF_mean2 = 0.0;
		
		// Calculate mean of the kernels
		for(int i = 0;i<(int)kerX;i++)
		{
			for(int j =0;j<(int)kerY;j++)
			{
				RF_mean1 += RF1[(start_idx+i-(int)halfX)*(int)samples + (start_idy+j-(int)halfY)];
				RF_mean2 += RF2[(start_idx+i-(int)halfX)*(int)samples + (start_idy+j-(int)halfY)];
			}
		}
		
		RF_mean1 = RF_mean1/(kerX*kerY);
		RF_mean2 = RF_mean2/(kerX*kerY);
		
		// Calculate NCC Value
		double numerator = 0.0;
		double denominator = 0.0;
		double den_sum1 = 0.0;
		double den_sum2 = 0.0;
		double cc = 0.0;
		
		for(int i = 0;i<(int)kerX;i++)
		{
			for(int j =0;j<(int)kerY;j++)
			{
				numerator += (RF1[(start_idx+i-(int)halfX)*(int)samples + (start_idy+j-(int)halfY)] - RF_mean1)*(RF2[(start_idx+i-(int)halfX)*(int)samples + (start_idy+j-(int)halfY)] - RF_mean2);
				
				den_sum1 += (RF1[(start_idx+i-(int)halfX)*(int)samples + (start_idy+j-(int)halfY)] - RF_mean1)*(RF1[(start_idx+i-(int)halfX)*(int)samples + (start_idy+j-(int)halfY)] - RF_mean1);
				
				den_sum2 += (RF2[(start_idx+i-(int)halfX)*(int)samples + (start_idy+j-(int)halfY)] - RF_mean2)*(RF2[(start_idx+i-(int)halfX)*(int)samples + (start_idy+j-(int)halfY)] - RF_mean2);
			}
		}
		
		denominator = sqrt(den_sum1*den_sum2);
		
		cc = numerator/denominator;
		NCC[idx*(int)numY + idy] = cc;
		
	}
}