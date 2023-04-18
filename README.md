# PSAP_PA_Beamforming

Repository containing codes for performing [Photoacoustic Sub-Aperture Beamforming](https://ieeexplore.ieee.org/abstract/document/9358181) published in IEEE Transactions on Ultrasonics, Ferroelectrics, and Frequency Control. An example of PSAP beamforming for photoacoustic inclusion phantom is shown below.

![An example of PSAP beamforming for photoacoustic inclusion phantom.](/assets/images/V1.png)

## Basic usage 

1. **PAI_Lesion_Perform_Beamform.m** demonstrates the usage of PSAP beamfroming using PA inclusion phantom channel data simulated with **k-Wave**.
2. The inclusion data can be found in **Lesion_Channel_Data** folder.
3. Codes were written to run on CUDA enabled GPU through MATLAB (To learn more , look here: https://www.mathworks.com/help/parallel-computing/run-cuda-or-ptx-code-on-gpu.html;jsessionid=8609102645505475695463f01b83).

### Issues or questions

Please reach out to Rashid Al Mukaddim, PhD at rashid102405@gmail.com with any concerns or questions.


