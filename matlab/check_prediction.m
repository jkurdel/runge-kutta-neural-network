% Script for testing 100-step prediction for set of 440 initial condition
% First run script gofr_van_der_pol.m to retrieve data!!!

close all;

MSE_x1 = [];
MSE_x2 = [];

n = 0;
centers = 0;
i_x1 = 0;

% calulate error for all 440 initial conditions
for x1 = -2:0.2:2
    i_x2 = 0;
    i_x1 = i_x1 + 1;
    for x2 = -2:0.2:2
        i_x2 = i_x2 + 1;
        n = n + 1
        centers(n,1) = x1;
        centers(n,2) = x2;
        if (x1 == 0 && x2 == 0)
            MSE2_x1(i_x1, i_x2) = 0;
            MSE2_x2(i_x1, i_x2) = 0;
            continue;
        end

    P = 100;        % prediction steps
    y_rbf1 = zeros(1,P);
    y_rbf2 = zeros(1,P);
    y_rbf1(1) = x1;
    y_rbf2(1) = x2;
    [t Y] = rk_van_der_pol(P*0.1-0.1, 0.1, [y_rbf1(1) y_rbf2(1)]);
    y1 = Y(:,1);
    y2 = Y(:,2);

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

    MSE_x1(i_x1, i_x2) = sum((y_rbf1(1:length(y1))' - y1).^2) / length(y1);
    MSE_x2(i_x1, i_x2) = sum((y_rbf2(1:length(y2))' - y2).^2) / length(y2);

    end
end

% show graphs with MSE for initital conditons
[t Y] = rk_van_der_pol(t_end, t_step, [0 2]);

figure(1)
pcolor(-2:0.2:2, -2:0.2:2, MSE_x1');
shading flat;
axis([-2.5 2.5 -3 3]);
hold on
plot(Y(:,1),Y(:,2),'g');
colorbar;

MSE_x2(find(MSE_x2 >= 10)) = 10;

figure(2)
pcolor(-2:0.2:2, -2:0.2:2, MSE_x2');
shading flat;
axis([-2.5 2.5 -3 3]);
hold on
plot(Y(:,1),Y(:,2),'g');
colorbar;