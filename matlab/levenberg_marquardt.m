% Levenberg-Marquardt method for approximating gaussian curve in form 
% f(x) = exp(-(x-c)^2/(2*sigma^2)) with added some noise.

close all;
clear all;
clc;

x = 0:0.1:20;

f = gaussmf(x, [2, 10]);
f = f + randn(size(f)) * 0.05;

plot(x, f, '*');
hold on;

sigma = 2;
c = 7;
w = 0.8;

Theta = [w; sigma; c];
y = Theta(1) * gaussmf(x, [Theta(2), Theta(3)]);

plot(x, y, 'r');

N = 1000;
for i = 1:N
    dy_dw = gaussmf(x, [Theta(2), Theta(3)]);
    dy_dsigma = Theta(1) * (x - Theta(3)).^2 / (Theta(2)^3) .* y;
    dy_dc = Theta(1) * (x - Theta(3)) / (Theta(2)^2) .* y;
    
    Z = [dy_dw' dy_dsigma' dy_dc'];
    e = f' - y';

    Theta = Theta + pinv(Z'*Z + 0.1*eye(3))*Z'*e;
    y = Theta(1) * gaussmf(x, [Theta(2), Theta(3)]);
end

plot(x, y, 'r');
