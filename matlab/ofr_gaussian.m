% Neural network with radial basis functions (RBF) approximating gaussian-based function 
% using Orthogonal Forward Regression (OFR)

close all;
clear all;
clc;

% ----- generating training data -----
x = 0:0.1:20;
sigma = 2;
c1 = 5;
c2 = 10;
c3 = 15;

y1 = gaussmf(x, [sigma c1]);
y2 = gaussmf(x, [sigma c2]);
y3 = gaussmf(x, [sigma c3]);
y = 2.5*y1 + 1.6*y2 - 7*y3;

% add noise to signal
y = y + randn(size(y)) * 0.05 * max(y);

plot(x, y, 'b');

N = 3;
[G centers sigmas] = generate_library_1d(x, N);
rbf_number = length(centers);

% TODO: comment these variables
D = y';
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
    Q_k(:,k) = Q(:,selected_rbfs(k));
    E = zeros(size(E));    
end

W = A_k\B_k';

% ----- function approximation by RBF neural network -----
figure(1)
hold on;

y_rbf = 0;
for i = 1:K
    y_rbf = y_rbf + W(i) * gaussmf(x, [sigmas(selected_rbfs(i)) centers(selected_rbfs(i))]);
end

plot(x, y_rbf, 'g');
hold off;

err = sum((y - y_rbf).^2);