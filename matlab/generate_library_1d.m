% ----- Function generating radial basis function set -----
% N - number of dividing set iteration - change this variable if want other number of RBFs
% G - matrix with values of all rbf functions
% centers - vector of centers of all rbf functions
% sigmas - vector of sigma of all rbf functions

function [G,centers,sigmas] = generate_library_1d(x, N)
    rbf_number = 0; % total number of RBFs

    for i = 1:N
        rbf_number = rbf_number + 2^i + 1;
    end

    G = zeros(length(x), rbf_number);    % matrix with values of all rbf functions
    centers = zeros(1, rbf_number);      % vector of centers of all rbf functions
    sigmas = zeros(1, rbf_number);       % vector of sigma of all rbf functions
    k = 1;

    for i = 1:N
        iter_rbf_count = 2^i+1;  % number of RBFs in current iteration
        sigma = sqrt(max(x) - min(x)) / 2^(i-1);

    %     figure(i+1)
    %     title({sprintf('%d level of RBF functions',i);
    %            sprintf('sigma = %f',sigma);
    %            sprintf('Functions count = %d',iter_rbf_count)})
    %     hold on

        for j = 1:iter_rbf_count
            sigmas(k) = sigma;
            % generating centers for rbf functions
            centers(k) = min(x) + (max(x) - min(x)) / (iter_rbf_count-1) * (j-1);
            G(:,k) = gaussmf(x, [sigma centers(k)]); 
    %         plot(x, G(:,k));
    %         plot(centers(k),0,'r*');
            k = k + 1;
        end    

    %     hold off;
    end
end