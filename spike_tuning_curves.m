%This file reads in a csv file that has all the spike times and is
%saved from nengo.  It generates and plots neuron tuning curves based
%on those spikes.  It assumes the spikes are generated from the sweeps
%of delta functions generated by this code, which must be run after the
%'noneuron' code to generate the basis:
%%%
% Code to generate the input signals for nengo to get spike trains
% slider = basis'*eye(length(x));
% slider = 15* slider./(ones(20,1)*sqrt(sum(slider.^2)));
% save('input_matrix.txt', 'slider', '-ascii')
%%%
clear all
spikes = csvread('outspikes.csv'); %spike times from selected neurons

%Do a spike count over successive 50ms windows?
dt = .001; %dt of data
T=16; %number of second of data

bin_size = .05; %Bin size in seconds

for i = 1:T/bin_size
    spike_count(:,i) = sum(spikes>(i-1)*bin_size & spikes<i*bin_size,2);
end

figure(10);clf;hold on;
n=[3 8 12 23 15 9 17 26 24 19]; %neurons to plot
dx = 1/size(spike_count,2);
x = [-1:2*dx:1-dx];
for i = n
    plot(x,1/bin_size*smooth(spike_count(i,:),10),'k')
end    

xlabel('x');
ylabel('Spike Rate (Hz)');
title('Sample Tuning Curves for Spiking Neurons');