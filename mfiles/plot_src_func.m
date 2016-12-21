%% plot src func
clear all
close all;

factor = 1;
f0 = 0.05;
t0 = 1.2 / f0;
t00 = 0;
tmax = 100;
it = 20;
% dt = 0.153846;
angle = 90;
dtr = 0.017454;


t = linspace(t00,tmax,it);
a = pi^2 * f0^2;

source = factor * exp(-a * (t-t0).^2);
% source = - factor * 2.d0*a*(t-t0).*exp(-a*(t-t0).^2);
plot(t,source);