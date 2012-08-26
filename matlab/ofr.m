function [selected_rbfs, W, E_k, A_k, Q_k, B_k] =  ofr(y, G, centers, K)

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

end