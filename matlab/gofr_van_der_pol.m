% Neural network with radial basis functions (RBF) approximating 
% Van der Pol equation using Orthogonal Generalized Orthogonal Forward 
% Regression (OFR)

close all;
clear all;
clc;

single_trajectory = true; % switch between training based on one trajectory or many trajectories

t_end = 40;         % time end of trajectory (always starts from time = 0)
t_step = 0.1;       % time step

x1 = 0;
x2 = 2;
if (single_trajectory)
else
    x1 = -2:0.5:2;      
    x2 = x1;
end

X = [];
y1 = [];
y2 = [];
for i = x1
    for j = x2
        if (i == 0 && j == 0)
            continue;
        end
        [t Y] = rk_van_der_pol(t_end, t_step, [i j]);
        X = [X; Y(1:end-1,:)];
        y1 = [y1; Y(2:end,1)];
        y2 = [y2; Y(2:end,2)];
    end
end

N = 3; % change number of library division (number of RBFs)
MAX_RBFS = 30; % change number of maximum possible selected RBFs

err = zeros(MAX_RBFS,2);
stop_condition = 1e-6;
if (single_trajectory)
else
    stop_condition = 1e-5;
end

% Training y(t) of Van der Pol equation
for K1 = 1:MAX_RBFS
    K1
    % ----- generating radial basis function set -----
    [G1 centers1 sigmas1] = generate_library_2d(X, N);
    % ----- teach RBF neural network -----
    [selected_rbfs1, W1, E_k1, A_k1, Q_k1, B_k1, centers1, sigmas1, G1] =  gofr(X, y1, G1, centers1, sigmas1, K1);
    y_rbf1 = 0;
    for i = 1:K1
        y_rbf1 = y_rbf1 + W1(i) * G1(:,selected_rbfs1(i));
    end
    
    % --- uncomment to show network after each selected RBF
%     figure(1)
%     plot(0:t_step:length(y1)*t_step-t_step, y1, 'r', 0:t_step:length(y_rbf1)*t_step-t_step, y_rbf1, 'b');
%     title({'y(t) of Van der Pol equation'; sprintf('Iteration nr = %d', K1)});
%     legend('desired','approximated');
%     pause;

    err(K1,1) = sum((y1 - y_rbf1).^2) / length(y1);
    if (err(K1,1) < stop_condition)
        break;
    end
end

% Training y'(t) of Van der Pol equation
for K2 = 1:MAX_RBFS
    K2
    [G2 centers2 sigmas2] = generate_library_2d(X, N);
    [selected_rbfs2, W2, E_k2, A_k2, Q_k2, B_k2, centers2, sigmas2, G2] =  gofr(X, y2, G2, centers2, sigmas2, K2);
    y_rbf2 = 0;
    for i = 1:K2
        y_rbf2 = y_rbf2 + W2(i) * G2(:,selected_rbfs2(i));
    end
    
    % --- uncomment to show network after each selected RBF
%     figure(1)
%     plot(0:t_step:length(y2)*t_step-t_step, y2, 'r', 0:t_step:length(y_rbf2)*t_step-t_step, y_rbf2, 'b');
%     title({'y''(t) of Van der Pol equation'; sprintf('Iteration nr = %d', K2)});
%     legend('desired','approximated');
%     pause;
    
    err(K2,2) = sum((y2 - y_rbf2).^2) / length(y2);
    if (err(K2,2) < stop_condition)
        break;
    end

end

% ----- function approximation by RBF neural network -----
if (single_trajectory)
    figure(1)
    plot(0:t_step:length(y1)*t_step-t_step, y1, 'r', 0:t_step:length(y_rbf1)*t_step-t_step, y_rbf1, 'b');
    title({'y(t) of Van der Pol equation'; sprintf('Iteration nr = %d', K1)});
    legend('desired','approximated');

    figure(2)
    plot(0:t_step:length(y2)*t_step-t_step, y2, 'r', 0:t_step:length(y_rbf2)*t_step-t_step, y_rbf2, 'b');
    title({'''y(t) of Van der Pol equation'; sprintf('Iteration nr = %d', K1)});
    legend('desired','approximated');
    
    figure(3)
    hold on;
    plot(y1, y2, 'r', y_rbf1, y_rbf2, 'b');
    title('Van der Pol equation');
    legend('desired','approximated');
end

% ----- function predictioin by RBF neural network -----
P = 100;        % prediction steps
y_rbf1 = zeros(1,P);
y_rbf2 = zeros(1,P);
y_rbf1(1) = 0;
y_rbf2(1) = 2;
[t Y] = rk_van_der_pol(P*0.1-0.1, 0.1, [y_rbf1(1) y_rbf2(1)]);
y1 = Y(:,1);
y2 = Y(:,2);

for j = 2:P
    for i = 1:K1
        y_rbf1(j) = y_rbf1(j) + W1(i) * gaussian_2D([y_rbf1(j-1) y_rbf2(j-1)], sigmas1(selected_rbfs1(i)), centers1(:,selected_rbfs1(i))');
    end
    
    for i = 1:K2
        y_rbf2(j) = y_rbf2(j) + W2(i) * gaussian_2D([y_rbf1(j-1) y_rbf2(j-1)], sigmas2(selected_rbfs2(i)), centers2(:,selected_rbfs2(i))');        
    end    
end

if (single_trajectory)
    figure(4)
    plot(0:t_step:length(y1)*t_step-t_step, y1, 'r', 0:t_step:length(y_rbf1)*t_step-t_step, y_rbf1, 'b');
    title({'y(t) of Van der Pol equation'; sprintf('Number of RBFs = %d', K1)});
    legend('desired','predicted');

    mse1_100 = sum((y_rbf1(1:length(y1))' - y1).^2) / length(y1);

    figure(5)
    plot(0:t_step:length(y2)*t_step-t_step, y2, 'r', 0:t_step:length(y_rbf2)*t_step-t_step, y_rbf2, 'b');
    title({'y''(t) of Van der Pol equation'; sprintf('Number of RBFs = %d', K2)});
    legend('desired','predicted');

    mse2_100 = sum((y_rbf2(1:length(y2))' - y2).^2) / length(y2);

    figure(6)
    plot(0:t_step:length(y1)*t_step-t_step,(y_rbf1(1:length(y1))' - y1),'r');
    title('y(t) of Van der Pol equation prediction error');

    figure(7)
    plot(0:t_step:length(y2)*t_step-t_step,(y_rbf2(1:length(y2))' - y2),'r');
    title('y''(t) of Van der Pol equation prediction error');

    % ----- function predictioin by RBF neural network -----
    P = 500;        % prediction steps
    y_rbf1 = zeros(1,P);
    y_rbf2 = zeros(1,P);
    y_rbf1(1) = 0;
    y_rbf2(1) = 2;
    [t Y] = rk_van_der_pol(P*0.1-0.1, 0.1, [y_rbf1(1) y_rbf2(1)]);
    y1 = Y(:,1);
    y2 = Y(:,2);


    for j = 2:P
        for i = 1:K1
            y_rbf1(j) = y_rbf1(j) + W1(i) * gaussian_2D([y_rbf1(j-1) y_rbf2(j-1)], sigmas1(selected_rbfs1(i)), centers1(:,selected_rbfs1(i))');
        end

        for i = 1:K2
            y_rbf2(j) = y_rbf2(j) + W2(i) * gaussian_2D([y_rbf1(j-1) y_rbf2(j-1)], sigmas2(selected_rbfs2(i)), centers2(:,selected_rbfs2(i))');        
        end    
    end

    if (single_trajectory)
        figure(8)
        plot(0:t_step:length(y1)*t_step-t_step, y1, 'r', 0:t_step:length(y_rbf1)*t_step-t_step, y_rbf1, 'b');
        title({'y(t) of Van der Pol equation'; sprintf('Number of RBFs = %d', K1)});
        legend('desired','predicted');

        mse1_500 = sum((y_rbf1(1:length(y1))' - y1).^2) / length(y1);

        figure(9)
        plot(0:t_step:length(y2)*t_step-t_step, y2, 'r', 0:t_step:length(y_rbf2)*t_step-t_step, y_rbf2, 'b');
        title({'y''(t) of Van der Pol equation'; sprintf('Number of RBFs = %d', K2)});
        legend('desired','predicted');

        mse2_500 = sum((y_rbf2(1:length(y2))' - y2).^2) / length(y2);

        figure(10)
        plot(0:t_step:length(y1)*t_step-t_step,(y_rbf1(1:length(y1))' - y1),'r');
        title('y(t) of Van der Pol equation prediction error');

        figure(11)
        plot(0:t_step:length(y2)*t_step-t_step,(y_rbf2(1:length(y2))' - y2),'r');
        title('y''(t) of Van der Pol equation prediction error');
    end
end