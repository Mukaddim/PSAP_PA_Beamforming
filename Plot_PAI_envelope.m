function [log_env_data] = Plot_PAI_envelope(data,depth,width,titlestr,DR)
% log_env_data = abs(data);
log_env_data = data;
% dispimg = abs(BfData_DAX_Ham);
log_env_data = log_env_data/max(log_env_data(:));
imagesc(width,depth,real(20*log10(log_env_data)),[DR 0]);
colormap hot;
title(titlestr);
xlabel('x (mm)');
ylabel('z (mm)');
% set(gca,'LineWidth',1.5,'FontSize',10,'FontName','FreeSans');
set(gca,'LineWidth',1.5,'FontSize',12,'FontName','FreeSans');
c=colorbar;
set(c,'LineWidth',2);
% c.Color = [1 1 1];
pbaspect([1 depth(end)/(2*width(end)) 1]);
% pbaspect([1 16/11.5 1]);
end

