function [selected_rbfs, W, E_k, A_k, Q_k, B_k, centers, sigmas, G] =  gofr(X, y, G, centers, sigmas, K)

    rbf_number = length(centers);

    % TODO: comment these variables
    D = y;
    Q = zeros(size(G));
    Q_k = zeros(size(G));
    B = zeros(1,rbf_number);
    B_k = zeros(1,rbf_number);
    E = zeros(1,rbf_number);
    E_k = zeros(1,K);
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
        W = A_k\B_k';

        % ------ Levenberg-Marquardt -----
        Theta = [W(k); sigmas(selected_rbfs(k)); centers(1, selected_rbfs(k)); centers(2, selected_rbfs(k))];

%         Theta_old = lsqnonlin(@rbf_mse,Theta,[],[],optimset('Algorithm','levenberg-marquardt','MaxFunEvals',100),X,y-y_rbf);

        y_rbf = 0;
        for j = 1:k
            y_rbf = y_rbf + W(j) * gaussian_2D(X, sigmas(selected_rbfs(j)), centers(:,selected_rbfs(j))');
        end
        

        lambda = 0.1;
        Theta_old = Theta;
        N = 100;   % maximum iterations of gradient method
        err = zeros(N,1);
        err(1) = abs(sum((y - y_rbf).^2)) / length(y);
        err_old = err(1);
        for n = 2:N
          
            dy_dw = gaussian_2d(X, Theta(2), [Theta(3) Theta(4)]);
            dy_dsigma = Theta(1) * ((Theta(3) - X(:,1)).^2 + (Theta(4) - X(:,2)).^2) ./ (Theta(2)^3) .* gaussian_2d(X, Theta(2), [Theta(3) Theta(4)]);
            dy_dc1 = (Theta(1)*(X(:,1) - Theta(3))) ./ (Theta(2)^2) .* gaussian_2d(X, Theta(2), [Theta(3) Theta(4)]);
            dy_dc2 = (Theta(1)*(X(:,2) - Theta(4))) ./ (Theta(2)^2) .* gaussian_2d(X, Theta(2), [Theta(3) Theta(4)]);
            
            Z = [dy_dw dy_dsigma dy_dc1 dy_dc2];
            e = y - y_rbf;

            Theta = Theta + pinv(Z'*Z + lambda*eye(4))*Z'*e;            
            W(k) = Theta(1);
            sigmas(selected_rbfs(k)) = Theta(2);
            centers(1,selected_rbfs(k)) = Theta(3);
            centers(2,selected_rbfs(k)) = Theta(4);
            
            y_rbf = 0;
            for j = 1:k
                y_rbf = y_rbf + W(j) * gaussian_2D(X, sigmas(selected_rbfs(j)), centers(:,selected_rbfs(j))');
            end

            err(n) = sum((y - y_rbf).^2) / length(y);
            if (err(n) >= err_old)
                Theta = Theta_old;
                lambda = lambda * 10;
            else
                Theta_old = Theta;
                err_old = err(n);
                lambda = lambda / 10;                
            end

            % if lambda is too big or too small stop
            if (lambda > 1e6 || lambda < 1e-6)
                break;
            end;
            
            if (err(n) < 1e-6)
                break;
            end
        end;
        
        % set the optimal parameters
        W(k) = Theta_old(1);
        sigmas(selected_rbfs(k)) = Theta_old(2);
        centers(1,selected_rbfs(k)) = Theta_old(3);
        centers(2,selected_rbfs(k)) = Theta_old(4);

        % Gram-Schmidt orthogonalization for modified RBF
        i = selected_rbfs(k);
        G(:,i) = gaussian_2D(X, Theta(2), [Theta(3) Theta(4)]);
        Q(:,i) = G(:,i);

        for j=1:k-1
            A{i}(j,k) = Q_k(:,j)'*G(:,i) / (Q_k(:,j)'*Q_k(:,j));
            Q(:,i) = Q(:,i) - A{i}(j,k)*Q_k(:,j);
        end

        B(i) = Q(:,i)'*D / (Q(:,i)'*Q(:,i));
        B_k(k) = B(i);
        E_k(k) = B(i)^2*Q(:,i)'*Q(:,i) / (D'*D);
        A_k(:,k) = A{i}(:,k);
        Q_k(:,k) = Q(:,i);

        W = A_k\B_k';

        E = zeros(size(E));
    end

    W = A_k\B_k';

end