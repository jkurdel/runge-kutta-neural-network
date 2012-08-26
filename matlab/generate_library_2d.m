% ----- Function generating radial basis function set -----
% N - number of dividing set iteration - change this variable if want other number of RBFs
% G - matrix with values of all rbf functions
% centers - vector of centers of all rbf functions
% sigmas - vector of sigma of all rbf functions

function [G,centers,sigmas] = generate_library_2d(X, N)

    rbf_number = 0; % total number of RBFs

    for i = 1:N
        rbf_number = rbf_number + (2^i + 1)^2;
    end

    G = zeros(length(X), rbf_number);
    centers = zeros(2, rbf_number);
    sigmas = zeros(1, rbf_number);
    k = 1;

    max_x = max(max(X));
    min_x = min(min(X));

    for i = 1:N

        iter_rbf_count = 2^i+1;  % number of RBFs in current iteration
        sigma = sqrt(max_x - min_x) / 2^(i-1);

    %     figure(i+1)
    %     title({sprintf('%d level of RBF functions',i);
    %            sprintf('sigma = %f',sigma);
    %            sprintf('Functions count = %d',iter_rbf_count^2)})
    %     hold on

        for j = 1:iter_rbf_count
            % generating centers for rbf functions
            for l = 1:iter_rbf_count
                centers(1, k) = min_x + (max_x - min_x) / (iter_rbf_count-1) * (j-1);
                centers(2, k) = min_x + (max_x - min_x) / (iter_rbf_count-1) * (l-1);
                G(:,k) = gaussian_2d(X, sigma, centers(:,k)'); 
    %             plot(centers(1,k),centers(2,k),'r*');
                sigmas(k) = sigma;
                k = k + 1;
            end
        end    

    %     hold off;
    end

end