% Neural network with radial basis functions (RBF) approximating first derivative 
% of Van der Pol equation using Generalized Orthogonal Forward Regression (GOFR)

close all;
clear all;
clc;

% ----- generating training data -----
[t Y] = rk_van_der_pol(50, 0.1, [0 2]);
y = Y(100:end,2);
X(:,1) = Y(98:end-2,2);
X(:,2) = Y(99:end-1,2);

plot(t(100:end), y, 'r');
title('y''(t) of Van der Pol equation');

% ----- generating radial basis function set -----
N = 3;
[G centers sigmas] = generate_library_2d(X, N);
rbf_number = length(centers);

% TODO: comment these variables
D = y;
Q = zeros(size(G));
Q_k = zeros(size(G));
B = zeros(1,rbf_number);
B_k = zeros(1,rbf_number);
E = zeros(1,rbf_number);
E_k = zeros(1,rbf_number);
selected_rbfs = zeros(1,rbf_number);      % indexes of selected rbfs order by decreasing energy 

A = cell(1,rbf_number);
A_k = eye(rbf_number);
for i = 1:rbf_number
    A{i} = eye(rbf_number);
end

% ----- selection of the most significant functions -----
K = 8;  % number of RBF selections
for k = 1:K
    % Gram-Schmidt orthogonalization
    for i = 1:rbf_number
       Q(:,i) = G(:,i);
       
       % if rbf is already selected function omit it 
       if ismember(i, selected_rbfs) == 1
           continue;
       end
       
       for j=1:k-1
            A{i}(j,k) = Q_k(:,j)'*G(:,i) / (Q_k(:,j)'*Q_k(:,j));
            Q(:,i) = Q(:,i) - A{i}(j,k)*Q_k(:,j);
       end
  
       B(i) = Q(:,i)'*D / (Q(:,i)'*Q(:,i));
       E(i) = B(i)^2*Q(:,i)'*Q(:,i) / (D'*D);   
    end
    
    % find RBF with maximum energy (save index and copy to Q_k matrix)
    [E_k(k) selected_rbfs(k)] = max(E);
    B_k(k) = B(selected_rbfs(k));
    A_k(:,k) = A{selected_rbfs(k)}(:,k);
    W = A_k\B_k';
    
    % ------ Levenberg-Marquardt -----
    Theta = [W(k); sigmas(selected_rbfs(k)); centers(1, selected_rbfs(k)); centers(2, selected_rbfs(k))];

    N = 1000;   % maximum iterations of gradient method
    n = 1;
    e_old = 1;
    e = 0;

    while (abs(sum(e_old.^2 - e.^2)) > 0.001) && (n < N)
        y_rbf = 0;
        for j = 1:k
            y_rbf = y_rbf + W(j) * gaussian_2D(X, sigmas(selected_rbfs(j)), centers(:,selected_rbfs(j))');
        end

        y_tmp = Theta(1) * gaussian_2D(X, Theta(2), [Theta(3) Theta(4)]);

        dy_dw = 1 ./ exp(((Theta(3) - X(:,1)).^2 + (Theta(4) - X(:,2)).^2) / (2*Theta(2)^2));
        dy_dsigma = Theta(1)*((Theta(3) - X(:,1)).^2 + (Theta(4) - X(:,2)).^2) ./ (Theta(2)^3*exp(((Theta(3) - X(:,1)).^2 + (Theta(4) - X(:,2)).^2)/(2*Theta(2)^2)));
        dy_dc1 = -(Theta(1)*(2*Theta(3) - 2*X(:,1))) ./ (2*Theta(2)^2*exp(((Theta(3) - X(:,1)).^2 + (Theta(4) - X(:,2)).^2)/(2*Theta(2)^2)));
        dy_dc2 = -(Theta(1)*(2*Theta(4) - 2*X(:,1))) ./ (2*Theta(2)^2*exp(((Theta(3) - X(:,1)).^2 + (Theta(4) - X(:,2)).^2)/(2*Theta(2)^2)));

        Z = [dy_dw dy_dsigma dy_dc1 dy_dc2];
        e_old = e;
        e = y - y_rbf;

        Theta = Theta + pinv(Z'*Z + 0.1*eye(4))*Z'*e;
        
        W(k) = Theta(1);
        sigmas(selected_rbfs(k)) = Theta(2);
        centers(1,selected_rbfs(k)) = Theta(3);
        centers(2,selected_rbfs(k)) = Theta(4);

        n = n + 1;
    end;
    
    % Gram-Schmidt orthogonalization for modified RBF
    i = selected_rbfs(k);
    G(:,i) = gaussian_2D(X, Theta(2), [Theta(3) Theta(4)]);
    Q(:,i) = G(:,i);
       
    for j=1:k-1
        A{i}(j,k) = Q_k(:,j)'*G(:,i) / (Q_k(:,j)'*Q_k(:,j));
        Q(:,i) = Q(:,i) - A{i}(j,k)*Q_k(:,j);
    end
  

    E_k(k) = B(i)^2*Q(:,i)'*Q(:,i) / (D'*D); 
    B_k(k) = Q(:,i)'*D / (Q(:,i)'*Q(:,i));
    A_k(:,k) = A{i}(:,k);
    Q_k(:,k) = Q(:,i);
    
    W = A_k\B_k';

    E = zeros(size(E));
    
    y_rbf = 0;
    for ij = 1:k
        y_rbf = y_rbf + W(ij) * gaussian_2D(X, sigmas(selected_rbfs(ij)), centers(:,selected_rbfs(ij))');
    end

    'blad sredniokwadratowy'
    err = sum((y - y_rbf).^2) / length(y)
end

W = A_k\B_k';


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

axis([t(200) t(200+P+1) -3 3]);
title({'y''(t) of Van der Pol equation';
       sprintf('Number of RBFs = %d', K)});
legend('desired','predicted');

hold off;