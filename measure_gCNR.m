function gCNR = measure_gCNR(target,background,plotf,filteron,min_h,max_h)
    
    % max_h = max([max(target(:)),max(background(:))]);
    % min_h = min([min(target(:)),min(background(:))]);
    
    % max_h = 0;
    % min_h = -65;

    [target_h,n] = histcounts(target,'NumBins',1000,'BinLimits',[min_h max_h],'Normalization','probability');
    background_h = histcounts(background,'NumBins',1000 , 'BinLimits',[min_h max_h],'Normalization','probability');
    
    % Median filtering
    if filteron
    target_h = medfilt1(target_h,3);
    background_h = medfilt1(background_h,3);
    end
    
    OVL = 0;
    for k = 1:1000
        OVL = OVL + min(target_h(k),background_h(k));
    end
    
    gCNR = 1 - OVL;
    
    if plotf
        figure,
        plot(n(1:end-1),target_h,'k','LineWidth',2);
        hold on;
        plot(n(1:end-1),background_h,'b','LineWidth',2);
        hold off;
        legend('Target','Background');
        ylabel('Probability');
        xlabel('Bins');
        set(gca,'LineWidth',2,'FontSize',16,'FontName','FreeSans');
    end
    
end

