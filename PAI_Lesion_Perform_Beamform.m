%% Perform DAS, CF and PSAP Beamforming on simulated inclusion channel data on k-Wave

clear all;
close all;


% Channel Data dir
input_dir = 'Lesion_Channel_Data/';  
if ~isfolder(input_dir)
    mkdir(input_dir);
end

% Beamformed Data dir
alternate_N = 2;
% res_dir = ['/data/data_us4/cardiac_mouse_model/ECG_Gated_PAI_2019/Simulation_Study_Apodization_Schemes/Diffuse_Lesion_Simulation_Data/PSAP_',num2str(alternate_N),'_',num2str(alternate_N),'_Beamformed_Data/'];

res_dir = ['PSAP_',num2str(alternate_N),'_',num2str(alternate_N),'_Beamformed_DataR/']; 
if ~isfolder(res_dir)
    mkdir(res_dir);
end

gfolder = 'altmany-export_fig-d8b9f4a/';
addpath(genpath(gfolder));
%% ----- Beamforming Parameter ----- %%

imagingwidth = 12e-3; % [m]
imagingdepth = 16e-3;   % [m]

pitch = 90e-6;                                              % LZ 250 pitch
TransducerElem = 128;                                       % number of transducer elements
f_c = 21e6;
imaging_aperture = pitch*TransducerElem;                    % aperture length for single quadrant imaging
imaging_aperture_half = imaging_aperture/2; 
sensor_location = linspace(-imaging_aperture_half,imaging_aperture_half,TransducerElem);

bf_param.Nele = 64;
% bf_param.Nele = 128;
bf_param.probe_name = 'LZ250';
bf_param.fs = 84e6;
bf_param.f_cen = f_c; % [MHz]
bf_param.BW = .55;
bf_param.DepthOffset = 0; % [m]
bf_param.ford = 128;
bf_param.multi_phase = 0;
bf_param.filter = 1;
bf_param.fiter_type = 2;
bf_param.fnum = 1;
bf_param.fnum_val = 1;
% --- Uniform Weighting --- %
bf_param.apeture_apo = 'uniform';
upsample_factor = [2 2];

width_axis = sensor_location*1e3;
dx = mean(diff(width_axis));


psap_results.das = cell(10,1);
psap_results.das_cf = cell(10,1);
psap_results.ncc = cell(10,1);
psap_results.dax_w = cell(10,1);
psap_results.phase = cell(10,1);
psap_results.phase_w = cell(10,1);
psap_results.psnr = zeros(10,4);
psap_results.ssim = zeros(10,4);

psap_results.snr_d1 = zeros(10,4);
psap_results.cr_d1 = zeros(10,4);
psap_results.gcnr_d1 = zeros(10,4);
psap_results.les_d1 = zeros(10,4);
psap_results.clut_d1 = zeros(10,4);

psap_results.snr_d2 = zeros(10,4);
psap_results.cr_d2 = zeros(10,4);
psap_results.gcnr_d2 = zeros(10,4);
psap_results.les_d2 = zeros(10,4);
psap_results.clut_d2 = zeros(10,4);


DR = -55;


for k = 1:10
    %% Load channel data
    load([input_dir,'ChannelData_',num2str(k),'.mat'],'channelcpu','gt_im');
    
    fprintf('Beamforming image number %d\n',k);
    
    filtereddata = gaussianFilter(channelcpu', 84e6, 21e6, 100, 0); % Apply bandlimited probe: Parameter same as our UFFC paper
    filtereddata = filtereddata'; 
    
    % DAS and DAS-CF Beamforming
    mask = ones(1,bf_param.Nele);
    bf_param.fnum = 1;
    bf_param.fnum_val = 1;
    [BfData_Uni_gpu,BfCfData_Uni_gpu,fs_int,Time_delayed_RF_gpu,CF_store_gpu] = BeamformPA_DAS_Simulation_GPU(filtereddata,bf_param,mask);
    
    psap_results.das{k} = BfData_Uni_gpu;
    psap_results.das_cf{k} = BfCfData_Uni_gpu;
    
    samples = size(BfData_Uni_gpu,1);
    depth_axis = linspace(0,imagingdepth,samples)*1e3;
    dy = mean(diff(depth_axis));

    % Sub Aperture Processing
    bf_param.fnum = 0;
    bf_param.fnum_val = 1;
    pattern = [ones(1,alternate_N) zeros(1,alternate_N)];
    pattern_r = repmat(pattern,[1 bf_param.Nele/length(pattern)]);
    [BfData_S,BfCfData_S,fs_int,Time_delayed_RF1,~] = BeamformPA_DAS_Simulation_GPU(filtereddata,bf_param,pattern_r);

    pattern = [zeros(1,alternate_N) ones(1,alternate_N)];
    pattern_r = repmat(pattern,[1 bf_param.Nele/length(pattern)]);
    [BfData_E,BfCfData_E,fs_int,Time_delayed_RF2,~] = BeamformPA_DAS_Simulation_GPU(filtereddata,bf_param,pattern_r);
    
    %% ---- DAX Coefficient Weighting with Amplitude Based NCC ---- %%
    % Caculate 2-D NCC
    c = 1540;
    lamda = (c/bf_param.f_cen)*1000; % mm
    
    dy_ncc = dy/upsample_factor(1);
    kernelY = round((1.5*lamda)/dy_ncc);
    kernelX = 3;
    ind=find(mod(kernelY,2)==0);
    kernelY(ind)=kernelY(ind)+1; % Making sure the kernel dimension is an odd number

    NCC_Alternate_gpu = gather(Zero_Lag_NCC_GPU(BfData_S,BfData_E,kernelY,kernelX,upsample_factor));

    
    BfData_Combine = BfData_Uni_gpu;
    DAX_weight = NCC_Alternate_gpu;
    DAX_weight(DAX_weight(:)<=0.001)=0.001;
    % DAX_weight = medfilt2(DAX_weight,[11 11]);
    DAX_weight = imresize(DAX_weight,size(BfData_Combine));
    BfData_DAX_Alternate = BfData_Combine.*DAX_weight;
    BfData_DAX_Alternate(~isfinite(BfData_DAX_Alternate))=0;
    
    psap_results.ncc{k} = BfData_DAX_Alternate;
    psap_results.dax_w{k} = DAX_weight;
    
    %% ---- Weighting with Phase-Based Cross-correlation Angle --- %

    [Time_delayed_RF1_IQ] = GenerateIQ_Data(Time_delayed_RF1);
    [Time_delayed_RF2_IQ] = GenerateIQ_Data(Time_delayed_RF2);
    IQ1 = sum(Time_delayed_RF1_IQ,3);
    IQ2 = sum(Time_delayed_RF2_IQ,3);
    IQ1 = upsample_data(IQ1,upsample_factor);
    IQ2 = upsample_data(IQ2,upsample_factor);
    complex_cc = IQ1.*conj(IQ2);
    phase_cc = angle(complex_cc); % When in-phase angle will be zero

    k0 = pi/3.5;
    weight_factor = exp(-phase_cc.^2/k0^2);
    weight_factor = medfilt2(weight_factor,[5 5]);
    weight_factor = imresize(weight_factor,size(BfData_Combine));
    BfData_DAX_NCC_angle = BfData_Combine.*weight_factor;
    
    psap_results.phase{k} = BfData_DAX_NCC_angle;
    psap_results.phase_w{k} = weight_factor;
    
    %% Ground Truth
    gt_im = imresize(gt_im,size(BfData_DAX_NCC_angle));
    
    
    %% ---- Plot the results
    h = figure;
    set(gcf,'units','inches','Position',[1 1 20 5]);
    subplot(1,5,1);
    titlestr = 'Ground Truth';
    GT_Env = gt_im/max(gt_im(:));
    % [GT_Env] = Plot_PAI_envelope(gt_im,depth_axis,width_axis,titlestr,DR);
    compute_metrics_lesion(width_axis,depth_axis,GT_Env,0,-2,2,7,1.5,1,DR,titlestr);
    
    
    
    subplot(1,5,2);
    titlestr = 'DAS';
    [DAS_Env] = Plot_PAI_envelope(envelope(BfData_Uni_gpu,32,'analytic'),depth_axis,width_axis,titlestr,DR);

    subplot(1,5,3);
    titlestr = 'DAS-CF';
    [DAS_CF_Env] = Plot_PAI_envelope(envelope(BfCfData_Uni_gpu,32,'analytic'),depth_axis,width_axis,titlestr,DR);
    
    subplot(1,5,4);
    titlestr = ['PSAP-',num2str(alternate_N),'-',num2str(alternate_N),' NCC'];
    [DAX_Env] = Plot_PAI_envelope(envelope(BfData_DAX_Alternate,32),depth_axis,width_axis,titlestr,DR);
    
    subplot(1,5,5);
    titlestr = ['PSAP-',num2str(alternate_N),'-',num2str(alternate_N),' Phase'];
    [DAX_Phase_Env] = Plot_PAI_envelope(envelope(BfData_DAX_NCC_angle,32),depth_axis,width_axis,titlestr,DR);
    
    im_name = [res_dir,'V_',num2str(k),'.tiff'];
%     export_fig(im_name,'-dtiff','-r300','-transparent');
    close(h);
    
    
    
    %% ----- Calculate pSNR and SSIM ----- %
    
    % Normalized envelope data
    
    
    psap_results.psnr(k,1) = psnr(GT_Env,DAS_Env);
    psap_results.psnr(k,2) = psnr(GT_Env,DAS_CF_Env);
    psap_results.psnr(k,3) = psnr(GT_Env,DAX_Env);
    psap_results.psnr(k,4) = psnr(GT_Env,DAX_Phase_Env);
    
    psap_results.ssim(k,1) = ssim(GT_Env,DAS_Env);
    psap_results.ssim(k,2) = ssim(GT_Env,DAS_CF_Env);
    psap_results.ssim(k,3) = ssim(GT_Env,DAX_Env);
    psap_results.ssim(k,4) = ssim(GT_Env,DAX_Phase_Env);

    % Derive Metrics for depth 1
    
    [psap_results.cr_d1(k,1),psap_results.snr_d1(k,1),psap_results.gcnr_d1(k,1),~,psap_results.les_d1(k,1),psap_results.clut_d1(k,1)] = ...
        compute_metrics_lesion(width_axis,depth_axis,DAS_Env,0,-2,2,7,1.5,0,DR,'DAS');
    
    [psap_results.cr_d1(k,2),psap_results.snr_d1(k,2),psap_results.gcnr_d1(k,2),~,psap_results.les_d1(k,2),psap_results.clut_d1(k,2)] = ...
        compute_metrics_lesion(width_axis,depth_axis,DAS_CF_Env,0,-2,2,7,1.5,0,DR,'DAS-CF');
    
    [psap_results.cr_d1(k,3),psap_results.snr_d1(k,3),psap_results.gcnr_d1(k,3),~,psap_results.les_d1(k,3),psap_results.clut_d1(k,3)] = ...
        compute_metrics_lesion(width_axis,depth_axis,DAX_Env,0,-2,2,7,1.5,0,DR,'DAX');
    
    [psap_results.cr_d1(k,4),psap_results.snr_d1(k,4),psap_results.gcnr_d1(k,4),~,psap_results.les_d1(k,4),psap_results.clut_d1(k,4)] = ...
        compute_metrics_lesion(width_axis,depth_axis,DAX_Phase_Env,0,-2,2,7,1.5,0,DR,'DAX-P');
    
    
    % Derive Metrics for depth 2
    
    [psap_results.cr_d2(k,1),psap_results.snr_d2(k,1),psap_results.gcnr_d2(k,1),~,psap_results.les_d2(k,1),psap_results.clut_d2(k,1)] = ...
        compute_metrics_lesion(width_axis,depth_axis,DAS_Env,0,-2,2,13,1.5,0,DR,'DAS');
    
    [psap_results.cr_d2(k,2),psap_results.snr_d2(k,2),psap_results.gcnr_d2(k,2),~,psap_results.les_d2(k,2),psap_results.clut_d2(k,2)] = ...
        compute_metrics_lesion(width_axis,depth_axis,DAS_CF_Env,0,-2,2,13,1.5,0,DR,'DAS-CF');
    
    [psap_results.cr_d2(k,3),psap_results.snr_d2(k,3),psap_results.gcnr_d2(k,3),~,psap_results.les_d2(k,3),psap_results.clut_d2(k,3)] = ...
        compute_metrics_lesion(width_axis,depth_axis,DAX_Env,0,-2,2,13,1.5,0,DR,'DAX');
    
    [psap_results.cr_d2(k,4),psap_results.snr_d2(k,4),psap_results.gcnr_d2(k,4),~,psap_results.les_d2(k,4),psap_results.clut_d2(k,4)] = ...
        compute_metrics_lesion(width_axis,depth_axis,DAX_Phase_Env,0,-2,2,13,1.5,0,DR,'DAX-P');
    
    
    

    

end


%% ----- Save results ----- %%

h = figure;
set(gcf,'units','inches','Position',[1 1 5 3]);
bh = boxplot(real(psap_results.cr_d1),'Widths',.4);
xticklabels({'DAS','DAS-CF','PSAP-NCC','PSAP-Phase'});
set(gca,'XTickLabelRotation',25,'FontName','FreeSans','FontSize',14,'LineWidth',1.5,'Tickdir','out');
set(bh,'LineWidth',1.5);
ylabel('CR (dB)');
xlabel('Methods');
title(sprintf('CR Comparsion: PSAP-%d-%d (7 mm)',alternate_N,alternate_N));
im_name = [res_dir,'CR_d1.tiff'];
export_fig(im_name,'-dtiff','-r300','-transparent');
close(h);


h = figure;
set(gcf,'units','inches','Position',[1 1 5 3]);
bh = boxplot(real(psap_results.cr_d2),'Widths',.4);
xticklabels({'DAS','DAS-CF','PSAP-NCC','PSAP-Phase'});
set(gca,'XTickLabelRotation',25,'FontName','FreeSans','FontSize',14,'LineWidth',1.5,'Tickdir','out');
set(bh,'LineWidth',1.5);
ylabel('CR (dB)');
xlabel('Methods');
title(sprintf('CR Comparsion: PSAP-%d-%d (13 mm)',alternate_N,alternate_N));
im_name = [res_dir,'CR_d2.tiff'];
export_fig(im_name,'-dtiff','-r300','-transparent');
close(h);

h = figure;
set(gcf,'units','inches','Position',[1 1 5 3]);
bh = boxplot(psap_results.gcnr_d1,'Widths',.4);
xticklabels({'DAS','DAS-CF','PSAP-NCC','PSAP-Phase'});
set(gca,'XTickLabelRotation',25,'FontName','FreeSans','FontSize',14,'LineWidth',1.5,'Tickdir','out');
set(bh,'LineWidth',1.5);
ylabel('gCNR');
xlabel('Methods');
title(sprintf('gCNR Comparsion: PSAP-%d-%d (7 mm)',alternate_N,alternate_N));
im_name = [res_dir,'gCNR_d1.tiff'];
export_fig(im_name,'-dtiff','-r300','-transparent');
close(h);

h = figure;
set(gcf,'units','inches','Position',[1 1 5 3]);
bh = boxplot(psap_results.gcnr_d2,'Widths',.4);
xticklabels({'DAS','DAS-CF','PSAP-NCC','PSAP-Phase'});
set(gca,'XTickLabelRotation',25,'FontName','FreeSans','FontSize',14,'LineWidth',1.5,'Tickdir','out');
set(bh,'LineWidth',1.5);
ylabel('gCNR');
xlabel('Methods');
title(sprintf('gCNR Comparsion: PSAP-%d-%d (13 mm)',alternate_N,alternate_N));
im_name = [res_dir,'gCNR_d2.tiff'];
export_fig(im_name,'-dtiff','-r300','-transparent');
close(h);

h = figure;
set(gcf,'units','inches','Position',[1 1 5 3]);
bh = boxplot(psap_results.snr_d1,'Widths',.4);
xticklabels({'DAS','DAS-CF','PSAP-NCC','PSAP-Phase'});
set(gca,'XTickLabelRotation',25,'FontName','FreeSans','FontSize',14,'LineWidth',1.5,'Tickdir','out');
set(bh,'LineWidth',1.5);
ylabel('SNR (dB)');
xlabel('Methods');
title(sprintf('SNR Comparsion: PSAP-%d-%d (7 mm)',alternate_N,alternate_N));
im_name = [res_dir,'SNR_d1.tiff'];
export_fig(im_name,'-dtiff','-r300','-transparent');
close(h);

h = figure;
set(gcf,'units','inches','Position',[1 1 5 3]);
bh = boxplot(psap_results.snr_d2,'Widths',.4);
xticklabels({'DAS','DAS-CF','PSAP-NCC','PSAP-Phase'});
set(gca,'XTickLabelRotation',25,'FontName','FreeSans','FontSize',14,'LineWidth',1.5,'Tickdir','out');
set(bh,'LineWidth',1.5);
ylabel('SNR (dB)');
xlabel('Methods');
title(sprintf('SNR Comparsion: PSAP-%d-%d (13 mm)',alternate_N,alternate_N));
im_name = [res_dir,'SNR_d2.tiff'];
export_fig(im_name,'-dtiff','-r300','-transparent');
close(h);


data_file = [res_dir,'PSAP_summery.mat'];
save(data_file,'psap_results');