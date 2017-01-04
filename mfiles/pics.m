%% make scalability plots

% 497 x 497 x 497
cores = [128 256 512 1024 2048 4096 8128 16384];
time4 = [259.94 1.370e+02 7.171e+01 3.955e+01 2.187e+01 1.419e+01 1.723e+01 5.105e+01];
timep = [259.94 259.94./[2 4 8 16 32 64 128]];
flops = [1.49e13  1.543e+13 1.596e+13 1.642e+13 1.638e+13  1.634e+13 1.634e+13 1.781e+13];
flopsec = [5.740e+10 1.126e+11 2.225e+11 4.152e+11 7.491e+11 1.151e+12 9.484e+11 3.489e+11];
mpimsg = [2.432e+03 5.120e+03 1.075e+04 2.202e+04 4.506e+04 9.216e+04 1.864e+05 3.768e+05];
mpilen = [8.277e+10 1.116e+11 1.422e+11 2.018e+11 2.577e+11 3.134e+11 4.270e+11 5.825e+11];


% subplot(4,2,5:8)
figure;
loglog(cores,time4,'r','LineWidth',4); hold on;
loglog(cores,timep,'b--','LineWidth',4);
NumTicks = 6;
% L = get(gca,'XLim');
% set(gca,'XTick',linspace(L(1),L(2),NumTicks))
% set(gca, 'XTickNumber', 11)
% set(gca, 'XTickLabel', cores)
xlabel('Processors');
ylabel('Time');
set(gca,'FontSize',14)
title('Time');
legend('Numerical', 'Perfect');

figure;
% subplot(4,2,1);
loglog(cores,flops,'b','LineWidth',4);
xlabel('Processors');
ylabel('Total Flops');
set(gca,'FontSize',14)
title('Flops');

figure;
% subplot(4,2,2);
loglog(cores,flopsec,'b','LineWidth',4);
xlabel('Processors');
ylabel('Flops/sec');
set(gca,'FontSize',14)
title('Flops/s');

figure;
% subplot(4,2,3);
loglog(cores,mpimsg,'b','LineWidth',4);
xlabel('Processors');
ylabel('MPI messages');
set(gca,'FontSize',14)
title('MPI msg');

figure;
% subplot(4,2,4);
loglog(cores, mpilen,'b','LineWidth',4);
xlabel('Processors');
ylabel('MPI messages length');
set(gca,'FontSize',14)
title('MPI msg len');

%%
% 997 x 997 x 997

cores = [128 256 512 1024 2048 4096 8128 16384];
time9 = [2.120e+03 1.015e+03 5.252e+02 2.857e+02 1.641e+02 1.052e+02 1.287e+02];
timep = [2.120e+03./[1 2 4 8 16 32 64]];
flops = [2.312e+14 2.311e+14 2.321e+14 2.441e+14 2.583e+14 2.618e+14 2.610e+14];
flopsec = [1.091e+11 2.278e+11 4.419e+11 8.546e+11 1.574e+12  2.489e+12 2.028e+12];
mpimsg = [5.120e+03 1.075e+04 2.202e+04 4.506e+04 9.216e+04 1.864e+05 3.768e+05];
mpilen = [8.406e+11 1.038e+12 1.441e+12 1.924e+12 2.462e+12 3.386e+12 4.274e+12];

figure;
loglog(cores(2:end),time9,'r','LineWidth',4); hold on;
loglog(cores(2:end),timep,'b--','LineWidth',4);
NumTicks = 6;
% L = get(gca,'XLim');
% set(gca,'XTick',linspace(L(1),L(2),NumTicks))
% set(gca, 'XTickNumber', 11)
% set(gca, 'XTickLabel', cores)
xlabel('Processors');
ylabel('Time');
set(gca,'FontSize',14)
title('Time');
legend('Numerical', 'Perfect');

figure;
% subplot(4,2,1);
loglog(cores(2:end),flops,'b','LineWidth',4);
xlabel('Processors');
ylabel('Total Flops');
set(gca,'FontSize',14)
title('Flops');

figure;
% subplot(4,2,2);
loglog(cores(2:end),flopsec,'b','LineWidth',4);
xlabel('Processors');
ylabel('Flops/sec');
set(gca,'FontSize',14)
title('Flops/s');

figure;
% subplot(4,2,3);
loglog(cores(2:end),mpimsg,'b','LineWidth',4);
xlabel('Processors');
ylabel('MPI messages');
set(gca,'FontSize',14)
title('MPI msg');

figure;
% subplot(4,2,4);
loglog(cores(2:end), mpilen,'b','LineWidth',4);
xlabel('Processors');
ylabel('MPI messages length');
set(gca,'FontSize',14)
title('MPI msg len');

%% 

loglog(cores,time4/max(time4),'b','LineWidth',4); hold on;
loglog(cores,[time9(1)*2 time9]/max([time9(1)*2 time9]),'r','LineWidth',4);
loglog(cores,[timep(1)*2 timep]/max([timep(1)*2 timep]),'k--','LineWidth',4);
xlabel('Processors');
ylabel('Time');
set(gca,'FontSize',14)
title('Strong scalability comparison');
legend('500^3', '1000^3','Perfect');

%% WEAK 32 - 997
cores = [1 4 16 64 256 1024];
timew = [1.260e+00 9.640e-01 4.695e+00 3.337e+01 1.359e+02 5.307e+02];
% timep = [2.120e+03./[1 2 4 8 16 32 64]];
flops = [1.940e+08 4.141e+09 6.639e+10 9.914e+11 1.543e+13 2.321e+14];
flopsec = [1.539e+08 4.295e+09 1.414e+10 2.971e+10 1.135e+11 4.374e+11];
mpimsg = [0 3.200e+01 2.240e+02 1.152e+03 5.120e+03 2.202e+04];
mpilen = [0 2.807e+07 5.600e+08 7.571e+09 1.116e+11 1.441e+12];

figure;
plot(cores,timew,'r','LineWidth',4); hold on;
% loglog(cores,timep,'b--','LineWidth',4);
xlabel('Processors');
ylabel('Time');
set(gca,'FontSize',14)
title('Time');
legend('Numerical', 'Perfect');

figure;
% subplot(4,2,1);
loglog(cores,flops,'b','LineWidth',4);
xlabel('Processors');
ylabel('Total Flops');
set(gca,'FontSize',14)
title('Flops');

figure;
% subplot(4,2,2);
loglog(cores,flopsec,'b','LineWidth',4);
xlabel('Processors');
ylabel('Flops/sec');
set(gca,'FontSize',14)
title('Flops/s');

figure;
% subplot(4,2,3);
loglog(cores,mpimsg,'b','LineWidth',4);
xlabel('Processors');
ylabel('MPI messages');
set(gca,'FontSize',14)
title('MPI msg');

figure;
% subplot(4,2,4);
loglog(cores, mpilen,'b','LineWidth',4);
xlabel('Processors');
ylabel('MPI messages length');
set(gca,'FontSize',14)
title('MPI msg len');
