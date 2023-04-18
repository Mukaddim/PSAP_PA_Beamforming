function [BfData,BfCfData,fs_int,Time_delayed_RF,CF_store] = BeamformPA_DAS_Simulation_GPU(sensordata,bf_param,mask)
%BeamformPA_DAS: DAS Beamforming for PAI for Simulation
% Updated Date: 06/19/2020
% Updates:
% 1. Output time delayed RF data before summation
% 2. Incorporated apodization : uniform, hamming
% 3. Sensor mask for allowing element turning on and off
% 4. Filering on and off option
% Note: Input sensorData should be at sampling frequency

% Note for GPU Implementation
% sensor data is a gpuArray either K-wave and defined as gpuArray variable

%%%%%%%%%%%%%%%%%%%%%%%%%%%% SETTINGS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Size of aperture to use for beamforming
NoElem=bf_param.Nele;
% NoElem=64;

% Set the region of the dataset to beamform (for speed)
StartLine = 0.0; % as a fractional percentage of total
EndLine = 1.0;  % as a fractional percentage of total
StartSample = 0.0; % as a fractional percentage of total
EndSample = 1.0; % as a fractional percentage of total

%constants
ct = 1540; %m/s
cl = 2340; %m/s

switch bf_param.probe_name
    case 'LZ250'
        % MS250/LZ250 settings
        a = 0.25e-3; %m - lens thickness
        pitch = 90e-6; %m
        
        
    case 'LZ400'
        % MS400/LZ400 settings
        a = 0.25e-3; %m - lens thickness
        pitch = 60e-6; %m
        
    otherwise
        % MS550/LZ550 settings
        a = 0.25e-3; %m - lens thickness
        pitch = 55e-6; %m
end

switch bf_param.apeture_apo
    case 'uniform'
        aperture_apo_w = ones(1,NoElem);
    case 'hamming'
        aperture_apo_w = hamming(NoElem);
    case 'blackman'
        aperture_apo_w = blackman(NoElem);
    case 'hann'
        aperture_apo_w = hann(NoElem);
%     case 'multi_phase'
%         
end

if bf_param.multi_phase
    m = bf_param.m;
    fo = bf_param.fo; % [cylces/mm]
    phase_type = bf_param.ph_type;
    x0 = [(-NoElem/2+0.5):(NoElem/2-1+0.5)]*pitch*1e3; % [mm]
    if strcmp(phase_type,'sin')
        phase_delay = 0.5*m*sin(2*pi*fo*x0);
    else
        phase_delay = 0.5*m*cos(2*pi*fo*x0);
    end    
end

samples = size(sensordata,1);
lines = size(sensordata,2);
DepthOffset = bf_param.DepthOffset; %mm
fs = bf_param.fs; %Hz

% Interpolation for beamforming
IntFac = 16;
fs_int = fs*IntFac;
FineDelayInc = 1/IntFac;

% Channel Data Filtering Options
f_center = bf_param.f_cen;
frac_BW = bf_param.BW;
f_low = f_center - (f_center*frac_BW)/2; % LZ 250 : 55% fractional Bandwidth
f_high = f_center + (f_center*frac_BW)/2;
f_pass = [f_low,f_high];
% f_ny = fs_int/2;
f_ny = fs/2;
w_pass = f_pass./f_ny;
filt_ord = bf_param.ford;
% bbp = fir1(filt_ord,w_pass);

if bf_param.fiter_type == 1 % Banpass filter
    Filt = designfilt('bandpassfir','FilterOrder',filt_ord, ...
         'CutoffFrequency1',f_low,'CutoffFrequency2',f_high, ...
         'SampleRate',fs);
    D = mean(grpdelay(Filt,1024));
elseif bf_param.fiter_type == 2 % FFT sampling High Pass
    f = [0 0.1 0.1 1]; 
    m = [0 0 1 1];
    DC_cancel = fir2(filt_ord,f,m);
else % Highpass filter
    Filt = designfilt('highpassfir','StopbandFrequency',w_pass(1), ... 
     'PassbandFrequency',w_pass(1)+.1,'PassbandRipple',0.5, ...
     'StopbandAttenuation',75,'DesignMethod','kaiserwin');
    D = mean(grpdelay(Filt,1024)); 
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% f = [0 0.1 0.1 1]; m = [0 0 1 1];
% DC_cancel = fir2(32,f,m);

% f = [0 0.1 0.1 1];
% mhi = [0 0 1 1];
% n = 64;
% win = tukeywin(n+1,0.5); %kaiser(n+1,0.5);
% bhi = fir2(n,f,mhi,win);

% Interpolate for RF Data generation
yq = linspace(1, samples, samples*IntFac); 
RfData = zeros(samples*IntFac,lines);
for i=1:lines
    % Interpolated RF
    if bf_param.filter
        if bf_param.fiter_type == 1 || bf_param.fiter_type == 3
            linedata = filter(Filt,[double(sensordata(:,i)); zeros(D,1)]); % Append D zeros to the input data
            linedata = linedata(D+1:end);                  % Shift data to compensate for delay
        else
            linedata = double(sensordata(:,i));
            linedata = conv(linedata, DC_cancel, 'same');
        end
        RfData(:,i) = interp1(linedata,yq); % Linear interpolation
    else
        RfData(:,i) = interp1(sensordata(:,i),yq); % Linear interpolation
    end
end


%% Delay-and-Sum Beamforming in CUDA Kernel

channel_dat = gpuArray(RfData);
Time_delayed_RF_gpu = gpuArray.zeros(samples,lines,NoElem); % Third dimension : time delayed aperture data for each pixel
aperture_apo_w_gpu = gpuArray(aperture_apo_w(:));
maskgpu = gpuArray(mask(:));
R = size(channel_dat,1);

numbeamformrow = length(floor(StartSample*samples)+1:floor(EndSample*samples));
BfData_gpu = gpuArray.zeros(numbeamformrow, lines);
BfCfData_gpu = gpuArray.zeros(numbeamformrow, lines);
CF_store_gpu = gpuArray.zeros(numbeamformrow, lines);

if ~bf_param.fnum
    kernel = parallel.gpu.CUDAKernel('DAS_beamform.ptx','DAS_beamform.cu','DAS_beamform');
    kernel.ThreadBlockSize = [128 2 1];
    kernel.GridSize = [1 round(numbeamformrow/2)+1 1];

    [BfData_gpu,BfCfData_gpu,Time_delayed_RF_gpu,CF_store_gpu] = feval(kernel,BfData_gpu,BfCfData_gpu,Time_delayed_RF_gpu,CF_store_gpu,channel_dat,aperture_apo_w_gpu,fs,...
        DepthOffset,ct,NoElem,pitch,maskgpu,FineDelayInc,IntFac,lines,numbeamformrow,R,a);

else
    fnum = bf_param.fnum_val;
    kernel = parallel.gpu.CUDAKernel('DAS_beamform_fnum.ptx','DAS_beamform_fnum.cu','DAS_beamform_fum');
    kernel.ThreadBlockSize = [128 2 1];
    kernel.GridSize = [1 round(numbeamformrow/2)+1 1];

    [BfData_gpu,BfCfData_gpu,Time_delayed_RF_gpu,CF_store_gpu] = feval(kernel,BfData_gpu,BfCfData_gpu,Time_delayed_RF_gpu,CF_store_gpu,channel_dat,aperture_apo_w_gpu,fs,...
        DepthOffset,ct,NoElem,pitch,maskgpu,FineDelayInc,IntFac,lines,numbeamformrow,R,a,fnum);
end
% kernel = parallel.gpu.CUDAKernel('beamform_DAS_GPU.ptx','beamform_DAS_GPU.cu','beamform_DAS_GPU');
% kernel.ThreadBlockSize = [128 2 1];
% kernel.GridSize = [1 round(numbeamformrow/2)+1 1];
% 
% [BfData_gpu,BfCfData_gpu,Time_delayed_RF_gpu,CF_store_gpu] = feval(kernel,BfData_gpu,BfCfData_gpu,Time_delayed_RF_gpu,CF_store_gpu,channel_dat,aperture_apo_w_gpu,fs,...
%     DepthOffset,ct,NoElem,pitch,maskgpu,FineDelayInc,IntFac,lines,numbeamformrow,R);
 

BfData = gather(BfData_gpu);
BfCfData = gather(BfCfData_gpu);
CF_store = gather(CF_store_gpu);
Time_delayed_RF = gather(Time_delayed_RF_gpu);

end


