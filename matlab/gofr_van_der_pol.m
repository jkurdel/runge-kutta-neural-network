% Neural network with radial basis functions (RBF) approximating first derivative 
% of Van der Pol equation using Generalized Orthogonal Forward Regression (GOFR)

close all;
clear all;
clc;

% ----- generating training data -----
[t Y] = rk_van_der_pol(100, 0.1, [0 2]);
y1 = Y(100:end,1);
X1(:,1) = Y(98:end-2,1);
X1(:,2) = Y(99:end-1,1);

y2 = Y(100:end,2);
X2(:,1) = Y(98:end-2,2);
X2(:,2) = Y(99:end-1,2);

% ----- generating radial basis function set -----
N = 2;
[G1 centers1 sigmas1] = generate_library_2d(X1, N);
[G2 centers2 sigmas2] = generate_library_2d(X2, N);

K = 8;
[selected_rbfs1, W1, E_k1, A_k1, Q_k1, B_k1, centers1, sigmas1] =  gofr(X1, y1, G1, centers1, sigmas1, K);
[selected_rbfs2, W2, E_k2, A_k2, Q_k2, B_k2, centers2, sigmas2] =  gofr(X2, y2, G2, centers2, sigmas2, K);

% ----- function approximation by RBF neural network -----
y_rbf1 = 0;
y_rbf2 = 0;
for i = 1:K
    y_rbf1 = y_rbf1 + W1(i) * gaussian_2D(X1, sigmas1(selected_rbfs1(i)), centers1(:,selected_rbfs1(i))');
    y_rbf2 = y_rbf2 + W2(i) * gaussian_2D(X2, sigmas2(selected_rbfs2(i)), centers2(:,selected_rbfs2(i))');
end
err1 = sum((y1 - y_rbf1).^2) / length(y1);
err2 = sum((y2 - y_rbf2).^2) / length(y2);

figure(1)
plot(t(100:end), y1, 'r', t(100:end), y_rbf1, 'b');
title('y(t) of Van der Pol equation');
legend('desired','approximated');

figure(2)
plot(t(100:end), y2, 'r', t(100:end), y_rbf2, 'b');
title('y''(t) of Van der Pol equation');
legend('desired','approximated');

figure(3)
hold on;
plot(y1, y2, 'r', y_rbf1, y_rbf2, 'b');
title('Van der Pol equation');
legend('desired','approximated');

% ----- function predictioin by RBF neural network -----
P = 200;        % prediction steps
y_rbf1 = zeros(1,P);
y_rbf2 = zeros(1,P);
y_rbf1(1:2) = y1(200:201);
y_rbf2(1:2) = y2(200:201);

for j = 3:P
    for i = 1:K
        y_rbf1(j) = y_rbf1(j) + W1(i) * gaussian_2D([y_rbf1(j-2) y_rbf1(j-1)], sigmas1(selected_rbfs1(i)), centers1(:,selected_rbfs1(i))');
        y_rbf2(j) = y_rbf2(j) + W2(i) * gaussian_2D([y_rbf2(j-2) y_rbf2(j-1)], sigmas2(selected_rbfs2(i)), centers2(:,selected_rbfs2(i))');
    end
end

figure(4)
plot(t(200:200+P-1),y1(200:200+P-1),'r', t(200:200+P-1),y_rbf1,'b');
axis([t(200) t(200+P+1) -3 3]);
title({'y(t) of Van der Pol equation'; sprintf('Number of RBFs = %d', K)});
legend('desired','predicted');

figure(5)
plot(t(200:200+P-1),y2(200:200+P-1),'r', t(200:200+P-1),y_rbf2,'b');
axis([t(200) t(200+P+1) -3 3]);
title({'y''(t) of Van der Pol equation'; sprintf('Number of RBFs = %d', K)});
legend('desired','predicted');