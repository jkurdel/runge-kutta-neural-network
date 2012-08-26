% Neural network with radial basis functions (RBF) approximating first derivative 
% of Van der Pol equation using Generalized Orthogonal Forward Regression (GOFR)

close all;
clear all;
clc;

% ----- generating training data -----
[t Y] = rk_van_der_pol(100, 0.1, [0 2]);
y = Y(100:end,2);
X(:,1) = Y(98:end-2,2);
X(:,2) = Y(99:end-1,2);

plot(t(100:end), y, 'r');
title('y''(t) of Van der Pol equation');

% ----- generating radial basis function set -----
N = 2;
[G centers sigmas] = generate_library_2d(X, N);
rbf_number = length(centers);

K = 9;  % number of RBF selections
[selected_rbfs, W, E_k, A, Q, B] =  ofr(y, G, centers, K);


% ----- function approximation by RBF neural network -----
figure(1)
hold on;

y_rbf = 0;
for i = 1:K
    y_rbf = y_rbf + W(i) * gaussian_2D(X, sigmas(selected_rbfs(i)), centers(:,selected_rbfs(i))');
end

plot(t(100:end), y_rbf, 'b');
legend('desired','approximated');
hold off;

err = sum((y - y_rbf).^2) / length(y);

P = 200;        % prediction steps
y_rbf = zeros(1,P);

y_rbf(1) = y(200);
y_rbf(2) = y(201);

for j = 3:P
    for i = 1:K
        y_rbf(j) = y_rbf(j) + W(i) * gaussian_2D([y_rbf(j-2) y_rbf(j-1)], sigmas(selected_rbfs(i)), centers(:,selected_rbfs(i))');
    end
end

figure(2)
hold on;
plot(t(200:200+P-1),y(200:200+P-1),'r');
plot(t(200:200+P-1),y_rbf,'b');

title({'y''(t) of Van der Pol equation'});
legend('desired','predicted');

hold off;