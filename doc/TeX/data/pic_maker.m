% Make pics

close all; 
clear all;

%% CFL 1
DP1025610 = dlmread('P10hz256PPWL10.txt');

figure;
plot(DP1025610/max(DP1025610),'b','LineWidth',2);
title('Imp. 10 Hz, 256x256, 10 PPWL')
ylabel('Amplitude');
xlabel('Samples');
set(gca,'FontSize',14)
axis([1 inf -1 1]);

print('../pics/DP1025610','-depsc');

% 122 s
%% CFL 0.5

SP1025610 = dlmread('S10hz256PPWL10.txt');

figure;
plot(SP1025610/max(SP1025610),'b','LineWidth',2);
title('Exp. 10 Hz, 256x256, 10 PPWL')
ylabel('Amplitude');
xlabel('Samples');
set(gca,'FontSize',14)
axis([1 inf -1 1]);

print('../pics/SP1025610','-depsc');

% 35 s
%%
DP2025605 = dlmread('P20hz256PPWL10.txt');

figure;
plot(DP2025605/max(DP2025605),'b','LineWidth',2);
title('Imp. 20 Hz, 256x256, 5 PPWL')
ylabel('Amplitude');
xlabel('Samples');
set(gca,'FontSize',14)
axis([1 inf -1 1]);

print('../pics/DP2025605','-depsc');

%127 s

%%
SP2025605 = dlmread('S20hz256PPWL10.txt');

figure;
plot(SP2025605/max(SP2025605),'b','LineWidth',2);
title('Exp. 20 Hz, 256x256, 5 PPWL')
ylabel('Amplitude');
xlabel('Samples');
set(gca,'FontSize',14)
axis([1 inf -1 1]);

print('../pics/SP2025605','-depsc');


%35 s

%% CFL 0.5

SP2051210 = dlmread('S20hz512PPWL10.txt');

figure;
plot(SP2051210/max(SP2051210),'b','LineWidth',2);
title('Exp. 20 Hz, 512x512, 10 PPWL')
ylabel('Amplitude');
xlabel('Samples');
set(gca,'FontSize',14)
axis([1 inf -1 1]);

print('../pics/SP2051210','-depsc');

%75 s

%% CFL 1
DP2012825 = dlmread('P20hz128PPWL2_5.txt');

figure;
plot(DP2012825/max(DP2012825),'b','LineWidth',2);
title('Imp. 20 Hz, 128x128, 2.5 PPWL')
ylabel('Amplitude');
xlabel('Samples');
set(gca,'FontSize',14)
axis([1 inf -1 1]);

print('../pics/DP2012825','-depsc');

% 14 s

%% CFL 1
DP2051210 = dlmread('P20hz512PPWL10.txt');

figure;
plot(DP2051210/max(DP2051210),'b','LineWidth',2);
title('Imp. 20 Hz, 512x512, 10 PPWL')
ylabel('Amplitude');
xlabel('Samples');
set(gca,'FontSize',14)
axis([1 inf -1 1]);

print('../pics/DP2051210','-depsc');

% 14 s

%% MAKE scaling TESTS
%STRONG
% 497 x 497 x 497
cores = [128 256 512 1024 2048 4096 8192 16384];
%Implicit
time4 = [259.94 1.370e+02 7.171e+01 3.955e+01 2.187e+01 1.419e+01 1.723e+01 5.105e+01];
% Explicit
timesf = [205.2 102.6 54.22 27.72 15.03 8.92 10.5 4.45];
%Perfect implicit
timep = [259.94 259.94./[2 4 8 16 32 64 128]];
%Perfect explicit
timepsf = [205.2 205.2 ./[2 4 8 16 32 64 128]];

% All the rest for implicit
flops = [1.49e13  1.543e+13 1.596e+13 1.642e+13 1.638e+13  1.634e+13 1.634e+13 1.781e+13];
flopsec = [5.740e+10 1.126e+11 2.225e+11 4.152e+11 7.491e+11 1.151e+12 9.484e+11 3.489e+11];
mpimsg = [2.432e+03 5.120e+03 1.075e+04 2.202e+04 4.506e+04 9.216e+04 1.864e+05 3.768e+05];
mpilen = [8.277e+10 1.116e+11 1.422e+11 2.018e+11 2.577e+11 3.134e+11 4.270e+11 5.825e+11];

% Imp 512 strong
figure;
loglog(cores,time4,'r','LineWidth',4); hold on;
h1 = loglog(cores,timep,'k--','LineWidth',2); %alpha(h1, 0.1); hold on;
xlabel('Processors');
ylabel('Time');
set(gca,'FontSize',14)
title('Strong scaling study 512^3. Implicit method');
legend('Imp. 512^3', 'Perfect Imp.');
print('../pics/imp512','-depsc');

% Exp 1024 strong
figure;
loglog(cores,timesf,'r','LineWidth',4); hold on;
loglog(cores,timepsf,'k--','LineWidth',2); %alpha(h2, 0.1);
xlabel('Processors');
ylabel('Time');
set(gca,'FontSize',14)
title('Strong scaling study 1024^3. Explicit method');
legend('Exp. 1024^3', 'Perfect Exp.');
print('../pics/exp1024','-depsc');

% Exp 2048 strong
figure;
timesf2048 = [3200 1609.9 821.77 420.12 214.45 118.71 66.89 37.19];
timepsf = [3200 3200./[2 4 8 16 32 64 128]];
loglog(cores,timesf2048,'r','LineWidth',4); hold on;
loglog(cores,timepsf,'k--','LineWidth',2); %alpha(h2, 0.1);
xlabel('Processors');
ylabel('Time');
set(gca,'FontSize',14)
title('Strong scaling study 2048^3. Explicit method');
legend('Exp. 2048^3', 'Perfect Exp.');
print('../pics/exp2048','-depsc');

%% Imp512 vs Exp1024
figure;
loglog(cores,time4,'r','LineWidth',4); hold on;
loglog(cores,timesf,'b','LineWidth',4); hold on;
xlabel('Processors');
ylabel('Time');
set(gca,'FontSize',14)
title('Strong scaling comparison. Coarse grid');
legend('Imp. 512^3', 'Exp. 1024^3');
print('../pics/imp512vsexp1024','-depsc');



%% Exp1024 vs Exp2048
timesf2048 = [3200 1609.9 821.77 420.12 214.45 118.71 66.89 37.19];
timepsf = [3200 3200./[2 4 8 16 32 64 128]];
figure;
loglog(cores,timesf/max(timesf),'b','LineWidth',4); hold on;
loglog(cores,timesf2048/max(timesf2048),'r','LineWidth',4); hold on;
loglog(cores,timepsf/max(timepsf),'k--','LineWidth',2); hold on;
xlabel('Processors');
ylabel('Time');
set(gca,'FontSize',14)
title('Strong scaling comparison 1024^3 vs 2048^3. Explicit method');
legend('Exp. 1024^3', 'Exp. 2048^3', 'Perfect');
print('../pics/exp1024vsexp2048','-depsc');

%%
figure('units','normalized','outerposition',[0 0 1 1]);
subplot(4,2,6:8);
loglog(cores,time4,'r','LineWidth',4); hold on;
h1 = loglog(cores,timep,'k--','LineWidth',2); %alpha(h1, 0.1); hold on;
xlabel('Processors');
ylabel('Time');
set(gca,'FontSize',14)
title('Strong scaling study 512^3. Implicit method');
legend('Imp. 512^3', 'Perfect Imp.');

% figure;
subplot(4,2,1);
loglog(cores,flops,'b','LineWidth',4);
xlabel('Processors');
ylabel('Total Flops');
set(gca,'FontSize',14)
title('Flops');

% figure;
subplot(4,2,2);
loglog(cores,flopsec,'b','LineWidth',4);
xlabel('Processors');
ylabel('Flops/sec');
set(gca,'FontSize',14)
title('Flops/sec');

% figure;
subplot(4,2,3);
loglog(cores,mpimsg,'b','LineWidth',4);
xlabel('Processors');
ylabel('MPI messages');
set(gca,'FontSize',14)
title('MPI msg');

% figure;
subplot(4,2,4);
loglog(cores, mpilen,'b','LineWidth',4);
xlabel('Processors');
ylabel('MPI messages length');
set(gca,'FontSize',14)
title('MPI msg len');


print('../pics/imp512strong','-depsc');

%%
% 997 x 997 x 997

cores = [128 256 512 1024 2048 4096 8192 16384];
time9 = [2.120e+03 1.015e+03 5.252e+02 2.857e+02 1.641e+02 1.052e+02 1.287e+02];
timep = [2.120e+03./[1 2 4 8 16 32 64]];
flops = [2.312e+14 2.311e+14 2.321e+14 2.441e+14 2.583e+14 2.618e+14 2.610e+14];
flopsec = [1.091e+11 2.278e+11 4.419e+11 8.546e+11 1.574e+12  2.489e+12 2.028e+12];
mpimsg = [5.120e+03 1.075e+04 2.202e+04 4.506e+04 9.216e+04 1.864e+05 3.768e+05];
mpilen = [8.406e+11 1.038e+12 1.441e+12 1.924e+12 2.462e+12 3.386e+12 4.274e+12];

%% Imp512 vs Imp1024

% figure('units','normalized','outerposition',[0 0 1 1]);
figure;
loglog(cores,time4/max(time4),'b','LineWidth',4); hold on;
loglog(cores,[time9(1)*2 time9]/max([time9(1)*2 time9]),'r','LineWidth',4);
loglog(cores,[timep(1)*2 timep]/max([timep(1)*2 timep]),'k--','LineWidth',2);
xlabel('Processors');
ylabel('Time');
set(gca,'FontSize',14)
title('Strong scaling comparison, 512^3 vs 1024^3. Implicit method');
legend('512^3', '1024^3','Perfect');

print('../pics/imp512vs1024strong','-depsc');

% %% Exp1024 vs Exp2048
% 
% % figure('units','normalized','outerposition',[0 0 1 1]);
% figure;
% loglog(cores,timesf2048/max(timesf2048),'b','LineWidth',4); hold on;
% loglog(cores,timesf/max(timesf),'r','LineWidth',4);
% % loglog(cores,[timep(1)*2 timep]/max([timep(1)*2 timep]),'k--','LineWidth',2);
% xlabel('Processors');
% ylabel('Time');
% set(gca,'FontSize',14)
% title('Strong scaling comparison, 1024^3 vs 2048^3. Explicit method');
% legend('1024^3', '2048^3','Perfect');
% 
% print('../pics/exp1024vs2048strong','-depsc');
% input('3');

%%
% Imp 1024 strong
figure;
loglog(cores,[time9(1)*2 time9],'r','LineWidth',4); hold on;
loglog(cores,[timep(1)*2 timep],'k--','LineWidth',2); %alpha(h1, 0.1); hold on;
xlabel('Processors');
ylabel('Time');
set(gca,'FontSize',14)
title('Strong scaling study 1024^3. Implicit method');
legend('Imp. 1024^3', 'Perfect Imp.');
print('../pics/imp1024','-depsc');


 %% Imp1024 vs Exp2048
figure;
loglog(cores,[time9(1)*2 time9],'r','LineWidth',4); hold on;
loglog(cores,timesf2048,'b','LineWidth',4); hold on;
xlabel('Processors');
ylabel('Time');
set(gca,'FontSize',14)
title('Strong scaling comparison. Fine grid');
legend('Imp. 1024^3', 'Exp. 2048^3');
print('../pics/imp1024vsexp2048','-depsc');


%%

figure('units','normalized','outerposition',[0 0 1 1]);
subplot(4,2,6:8);
loglog(cores(2:end),time9,'r','LineWidth',4); hold on;
loglog(cores(2:end),timep,'k--','LineWidth',2);
xlabel('Processors');
ylabel('Time');
set(gca,'FontSize',14)
title('Strong scaling study 1024^3. Implicit method');
legend('Imp. 1024^3', 'Perfect Imp.');

% figure;
subplot(4,2,1);
loglog(cores(2:end),flops,'b','LineWidth',4);
xlabel('Processors');
ylabel('Total Flops');
set(gca,'FontSize',14)
title('Flops');

% figure;
subplot(4,2,2);
loglog(cores(2:end),flopsec,'b','LineWidth',4);
xlabel('Processors');
ylabel('Flops/sec');
set(gca,'FontSize',14)
title('Flops/s');

% figure;
subplot(4,2,3);
loglog(cores(2:end),mpimsg,'b','LineWidth',4);
xlabel('Processors');
ylabel('MPI messages');
set(gca,'FontSize',14)
title('MPI msg');

% figure;
subplot(4,2,4);
loglog(cores(2:end), mpilen,'b','LineWidth',4);
xlabel('Processors');
ylabel('MPI messages length');
set(gca,'FontSize',14)
title('MPI msg len');

print('../pics/imp1024strong','-depsc');



%% WEAK 32 - 997
cores = [1 8 64 512 4096 32768];
timew = [7.862e-01 6.452e-01 2.171e+00 4.769e+00 1.629e+01 1.474e+02];
flops = [1.940e+08 4.263e+09 6.601e+10 1.030e+12 1.634e+13 2.604e+14];
flopsec = [2.467e+08 6.607e+09 3.041e+10 2.161e+11 1.003e+12 1.767e+12];
mpimsg = [0e+00 9.600e+01 1.152e+03 1.075e+04 9.216e+04 7.619e+05];
mpilen = [0e+00 4.325e+07 1.008e+09 1.842e+10 3.134e+11 5.163e+12];

figure('units','normalized','outerposition',[0 0 1 1]);
subplot(4,2,6:8);
plot(cores,timew,'r','LineWidth',4); hold on;
plot(cores,repmat(timew(1),1,length(cores)),'b--','LineWidth',4); hold on;
% loglog(cores,timep,'b--','LineWidth',4);
xlabel('Processors');
ylabel('Time');
set(gca,'FontSize',14)
title('Weak scaling study from 32^3 to 1024^3. Implicit method');
legend('Numerical', 'Perfect');

% figure;
subplot(4,2,1);
loglog(cores,flops,'b','LineWidth',4);
xlabel('Processors');
ylabel('Total Flops');
set(gca,'FontSize',14)
title('Flops');

% figure;
subplot(4,2,2);
loglog(cores,flopsec,'b','LineWidth',4);
xlabel('Processors');
ylabel('Flops/sec');
set(gca,'FontSize',14)
title('Flops/s');

% figure;
subplot(4,2,3);
loglog(cores,mpimsg,'b','LineWidth',4);
xlabel('Processors');
ylabel('MPI messages');
set(gca,'FontSize',14)
title('MPI msg');

% figure;
subplot(4,2,4);
loglog(cores, mpilen,'b','LineWidth',4);
xlabel('Processors');
ylabel('MPI messages length');
set(gca,'FontSize',14)
title('MPI msg len');

print('../pics/imp32-1024weak','-depsc');

%%
close all
