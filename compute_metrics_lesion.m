function [cr,snr,gcnr,cnr,lesion_level,clutter_level] = compute_metrics_lesion(x1,z1,data,centx,centx1,centx2,centz,rad,showROIFlag,DR,titlestr)



oneSided = 0;

if ~exist('showROIFlag','var')
    showROIFlag = 0;
end

% data = data/max(data(:));

% find inclusion
[xx zz] = meshgrid(x1,z1);
incids = sqrt( (xx-centx).^2 + (zz-centz).^2 )<=rad;
if ~oneSided
    bckids1 = (sqrt( (xx-centx1).^2 + (zz-centz).^2 )<=rad) & ((xx-centx1)<0);
    bckids2 = (sqrt( (xx-centx2).^2 + (zz-centz).^2 )<=rad) & ((xx-centx2)>0);
else
    bckids = (sqrt( (xx-centx1).^2 + (zz-centz).^2 )<=rad);
end
inc = data(incids); inc = inc(:);
if ~oneSided
bck1 = data(bckids1); bck1 = bck1(:);
bck2 = data(bckids2); bck2 = bck2(:);
bck = [bck1; bck2];
else
bck = data(bckids); bck = bck(:);
end
if showROIFlag == 1
    dataTmp = data; dataTmp(incids) = nan; 
    if ~oneSided
        dataTmp(bckids1) = nan; dataTmp(bckids2) = nan;
    else
        dataTmp(bckids) = nan;
    end
    
    imagesc(x1,z1,real(20*log10(data)),[DR 0]);
    colormap hot;
    title(titlestr);
    xlabel('x (mm)');
    ylabel('z (mm)');
    set(gca,'LineWidth',1.5,'FontSize',12,'FontName','FreeSans');
    c=colorbar;
    set(c,'LineWidth',2);
    % c.Color = [1 1 1];
    pbaspect([1 z1(end)/(2*x1(end)) 1]);
    
    hold on;
    
    map = zeros(size(dataTmp));
    map(isnan(dataTmp)) = 1;
    B = bwboundaries(map);
    
    [r,c] = size(data);
    x_l = xx(sub2ind([r,c],B{1}(:,1),B{1}(:,2)));
    z_l = zz(sub2ind([r,c],B{1}(:,1),B{1}(:,2)));
    
    x_c = xx(sub2ind([r,c],B{2}(:,1),B{2}(:,2)));
    z_c = zz(sub2ind([r,c],B{2}(:,1),B{2}(:,2)));
    
    x_r = xx(sub2ind([r,c],B{3}(:,1),B{3}(:,2)));
    z_r = zz(sub2ind([r,c],B{3}(:,1),B{3}(:,2)));
    
%     plot(x_l,z_l,'g--','MarkerSize',3);
%     plot(x_c,z_c,'b--','MarkerSize',3);
%     plot(x_r,z_r,'g--','MarkerSize',3);
    
    plot(x_l,z_l,'w-','LineWidth',1.5);
    plot(x_c,z_c,'g-','LineWidth',1.5);
    plot(x_r,z_r,'w-','LineWidth',1.5);
%     figure
%     imagesc(x1,z1,data,[0 3]); colormap parula; axis image; colorbar
%     imageProcess
end

cnr = abs(mean(inc) - mean(bck))/(sqrt((std(inc))^2 + (std(bck))^2));
cr = 20*log10(mean(inc)/mean(bck));
snr = 20*log10(mean(inc)/std(bck));

min_h = min(abs(data(:)));
max_h = max(abs(data(:)));

gcnr = measure_gCNR(inc,bck,0,0,min_h,max_h);

lesion_level = 20*log10(mean(inc));
clutter_level = 20*log10(mean(bck));
end

