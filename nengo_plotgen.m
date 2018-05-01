%%%
%Code for generating plots based on nengo data (exported from
%nengo using the export to matlab function in the data viewer).
%It assumes that some variables (e.g. decoders) have already
%been generated by the 'noneurons' code in matlab.

load('Normalization_direct.mat'); %loads data and data_time
nengo_direct = data;
T = max(data_time);
Tlen = length(data_time);
dt = T/length(data_time);

load('spikerun_bias.mat');
nengo_bias = data;
load('spikerun_nobias.mat');
nengo_nobias = data;
T_spikes = max(data_time);
t=data_time;
Tlen_spikes=length(data_time);
T_skip = Tlen_spikes/Tlen;

startup = .01; %length of time input signal is presented
startup = startup/dt;

%Filter the spike data
t_psc = .05; %PSC time constant
t = t(1:ceil(5*t_psc/t(1)));
psc = exp(-t/t_psc);
psc = psc/sum(psc);
for i = 1:20
   tmp = conv(nengo_nobias(:,i)',psc);
   nengo_nobias(:,i) = tmp(1:Tlen_spikes)';
   tmp = conv(nengo_bias(:,i)',psc);
   nengo_bias(:,i) = tmp(1:Tlen_spikes)';
end

nengo_nobias = nengo_nobias(1:T_skip:end,:);
nengo_bias = nengo_bias(1:T_skip:end,:);

direct = nengo_direct*decoders;
nobias = nengo_nobias*decoders;
bias = nengo_bias*decoders;

direct_area = sum(direct')*dx;
nobias_area = sum(nobias')*dx;
bias_area = sum(bias')*dx;

plot_times = [100 200 2000-startup] + startup;

figure(5);clf;hold on;
plot(x,direct(plot_times(1),:),'-','LineWidth',2,'Color',[.5 .5 .5]);
plot(x,direct(plot_times(2:end),:),'-','LineWidth',2,'Color',[.5 .5 .5],'HandleVisibility','off');
plot(x,nobias(plot_times,:),'k-','LineWidth',1);
xlabel('x')
ylabel('\rho(x)')
legend('ideal','without bias');
title('Comparison of ideal and unbiased inference in a spiking network');
% figure(1);clf;hold on;
% plot(p_a1);
% plot(p_a2,'r:');

figure(6);clf;hold on;
plot(x,direct(plot_times(1),:),'-','LineWidth',2,'Color',[.5 .5 .5]);
plot(x,direct(plot_times(2:end),:),'-','LineWidth',2,'Color',[.5 .5 .5],'HandleVisibility','off');
plot(x,bias(plot_times,:),'k-','LineWidth',1);
xlabel('x')
ylabel('\rho(x)')
legend('ideal','with bias');
title('Comparison of ideal and biased inference in a spiking network');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Plot the change in area in the decoded space
%over time with and without bias
figure(4);clf;hold on;
plot(ideal_area,'-','LineWidth',2,'Color',[.5 .5 .5]);
plot(direct_area(startup:20:end),'k:','LineWidth',2);
plot(bias_area(startup:20:end),'k','LineWidth',1);
plot(nobias_area(startup:20:end),'k--','LineWidth',1);
title('Bias Affect on Density Area');
legend('ideal','ideal dynamics','spikes','spikes without bias');
xlabel('Iteration'); ylabel('Area');

