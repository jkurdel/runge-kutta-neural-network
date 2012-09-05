% Neural network with radial basis functions (RBF) approximating 
% Van der Pol equation using Orthogonal Forward Regression (OFR)

close all;
clear all;
clc;

% ----- generating training data -----
[t Y] = rk_van_der_pol(50, 0.1, [0 2]);
X(:,1) = Y(99:end-1,1);
X(:,2) = Y(99:end-1,2);
y1 = Y(100:end,1);
y2 = Y(100:end,2);

% ----- generating radial basis function set -----
N = 3;

% ----- teach RBF neural network -----
for K1 = 1:30
    K1
    [G1 centers1 sigmas1] = generate_library_2d(X, N);
    [selected_rbfs1, W1, E_k1, A_k1, Q_k1, B_k1, centers1, sigmas1, G1] =  gofr(X, y1, G1, centers1, sigmas1, K1);
    y_rbf1 = 0;
    for i = 1:K1
        y_rbf1 = y_rbf1 + W1(i) * G1(:,selected_rbfs1(i));
    end
    
%     figure(1)
%     plot(t(100:end), y1, 'r', t(100:end), y_rbf1, 'b');
%     title({'y(t) of Van der Pol equation'; sprintf('Iteration nr = %d', K1)});
%     legend('desired','approximated');
%     pause;
    
    err(K1,1) = sum((y1 - y_rbf1).^2) / length(y1);
    
    if (err(K1,1) < 1e-5)
        break;
    end
end

for K2 = 1:30
    K2
    [G2 centers2 sigmas2] = generate_library_2d(X, N);
    [selected_rbfs2, W2, E_k2, A_k2, Q_k2, B_k2, centers2, sigmas2, G2] =  gofr(X, y2, G2, centers2, sigmas2, K2);
    y_rbf2 = 0;
    for i = 1:K2
        y_rbf2 = y_rbf2 + W2(i) * G2(:,selected_rbfs2(i));
    %     y_rbf2 = y_rbf2 + W2(i) * gaussian_2D(X, sigmas2(selected_rbfs2(i)), centers2(:,selected_rbfs2(i))');
    end
    
%     figure(1)
%     plot(t(100:end), y2, 'r', t(100:end), y_rbf2, 'b');
%     title({'y''(t) of Van der Pol equation'; sprintf('Iteration nr = %d', K2)});
%     legend('desired','approximated');
%     pause;
    
    err(K2,2) = sum((y2 - y_rbf2).^2) / length(y2);

    if (err(K2,2) < 1e-5)
        break;
    end

end

% ----- function approximation by RBF neural network -----


figure(1)
plot(t(100:end), y1, 'r', t(100:end), y_rbf1, 'b');
title({'y(t) of Van der Pol equation'; sprintf('Iteration nr = %d', K1)});
legend('desired','approximated');

figure(2)
plot(t(100:end), y2, 'r', t(100:end), y_rbf2, 'b');
title({'y''(t) of Van der Pol equation'; sprintf('Iteration nr = %d', K2)});
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
% y_rbf1(1) = y1(200);
% y_rbf2(1) = y2(200);
y_rbf1(1) = 0;
y_rbf2(1) = 2;
[t Y] = rk_van_der_pol(P*0.1-0.1, 0.1, [y_rbf1(1) y_rbf2(1)]);
y1 = Y(100:end,1);
y2 = Y(100:end,2);

y_rbf1(1) = y1(1);
y_rbf2(1) = y2(1);

for j = 2:P
    for i = 1:K1
        y_rbf1(j) = y_rbf1(j) + W1(i) * gaussian_2D([y_rbf1(j-1) y_rbf2(j-1)], sigmas1(selected_rbfs1(i)), centers1(:,selected_rbfs1(i))');
    end
    
    for i = 1:K2
        y_rbf2(j) = y_rbf2(j) + W2(i) * gaussian_2D([y_rbf1(j-1) y_rbf2(j-1)], sigmas2(selected_rbfs2(i)), centers2(:,selected_rbfs2(i))');        
    end
end

figure(4)
plot(t(1:end-99),y1,'r', t(1:end-99) ,y_rbf1(1:end-99),'b');
title({'y(t) of Van der Pol equation'; sprintf('Number of RBFs = %d', K1)});
legend('desired','predicted');

mse1 = sum((y_rbf1(1:end-99)' - y1).^2) / length(y1);

figure(5)
plot(t(1:end-99),y2,'r', t(1:end-99) ,y_rbf2(1:end-99),'b');
title({'y''(t) of Van der Pol equation'; sprintf('Number of RBFs = %d', K2)});
legend('desired','predicted');

mse2 = sum((y_rbf1(1:end-99)' - y2).^2) / length(y2);

figure(6)
plot(t(1:end-99),y1 - y_rbf1(1:end-99)','r');
title('y(t) of Van der Pol equation prediction error');

figure(7)
plot(t(1:end-99),y2 - y_rbf2(1:end-99)','r');
title('y''(t) of Van der Pol equation prediction error');
