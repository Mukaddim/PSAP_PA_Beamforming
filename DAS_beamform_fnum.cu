__global__ void DAS_beamform_fum(double* RF,double* RF_Cf,double* TimeDelayedRf,double* Cf_factor,double* channeldata,double* aperture_apo_w,double fs,
                                 double DepthOffset,double ct,double NoElem,double pitch,double* mask,double FineDelayInc,double IntFac,double lines,double samples,double R,double a,double fnum)
                                 
{
	int idx = threadIdx.x + blockDim.x*blockIdx.x; // Loop over A-lines
	int idy = threadIdx.y + blockDim.y*blockIdx.y; // Loop over Samples
	
	if(idy<samples)
	{
		double sd = DepthOffset*1e-3 + (idy + 1)*(ct/fs); // Sample Depth
		
		double ElemSum = 0.0;
		double CountSum = 0.0;
		double imageWidth = (lines - 1)*pitch;
		double rfval = 0.0;
		double Cf_TopSum = 0.0;
		double Cf_BottomSum = 0.0;
		
		double ap_len = (sd/fnum)/pitch;
		double rad_len = round(ap_len/2) + 1;
		double cen_el = round(NoElem/2);
		
		int l_start = cen_el - rad_len;
		int l_end = cen_el + rad_len;
		
		if(l_start<0) l_start = 0;
		if(l_end>NoElem) l_end = NoElem;
		
		for(int k = l_start; k < (int)l_end; k++) // Loop over aperture
		{
			if(mask[k]>0.0){ // Aperture pattern to be utlized in sub-aperture imaging
			double pitchoff = (-NoElem/2 + (double)k + 0.5)*pitch; // element offset with respect to aperture center
			double ElemPos = ((double)idx + 0.5)*pitch + pitchoff; // element position with respect to the A-line under consideration
			
			if(ElemPos>=-1e-9 && ElemPos<=imageWidth)
			{
				double DelayDistance = sqrt( (pitchoff*pitchoff) + (sd+a)*(sd+a) ); // [m]
				double DelaySamp = (DelayDistance - (sd+a))*fs/ct;
											
				// Find coarse delay
				int CoarseDelay = (int)floor(DelaySamp) + idy; // idy = row index of depth

				// Find fine delay
				double DelaySampFrac = DelaySamp - floor(DelaySamp);
				int FineDelay = (int)round(DelaySampFrac/FineDelayInc);
						
				// Final Sample Delay for DAS 
				int sampleDelay = (int)(CoarseDelay)*IntFac  + (int)FineDelay; // row index of delayed signal in interpolated data
				
				if(sampleDelay>=0 && sampleDelay<R) // R = row dimension of interpolated data
				{
					int base_idx = idx - (int)(NoElem/2) + 1 + k;
					rfval = aperture_apo_w[k]*channeldata[base_idx*(int)R + sampleDelay];
					ElemSum = ElemSum + rfval;
					
					CountSum = CountSum + 1.0;
					
					// Store Time-Delayed Channel Data
					TimeDelayedRf[k*(int)samples*int(lines) + idx*(int)samples + idy] = rfval; 
					
					// Coherence Factor Caclulation
					Cf_TopSum = Cf_TopSum + rfval;
					Cf_BottomSum = Cf_BottomSum + fabs(rfval)*fabs(rfval);
				}
			}
			}
		}
		
		
		// After Looping over aperture data: update results 
		RF[idx*(int)samples + idy] = ElemSum/CountSum; // DAS
		Cf_factor[idx*(int)samples + idy] = (fabs(Cf_TopSum)*fabs(Cf_TopSum))/(CountSum*Cf_BottomSum); // Coherence Factor
		RF_Cf[idx*(int)samples + idy] = Cf_factor[idx*(int)samples + idy]*(ElemSum/CountSum);  // DAS - CF
	}
}