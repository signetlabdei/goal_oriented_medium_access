function [thresholds, reward_history] = iterated_best_response(cdf, psi, epsilon, values, init_thresholds, max_iter)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                         %
%                    function: iterated_best_response                     %
%           author: Federico Chiariotti (chiariot@dei.unipd.it)           %
%                             license: GPLv3                              %
%                                                                         %
%                                                                         %
%                                                                         %
% Computes the LIBRA thresholds for a given initial policy by running the %
% Iterated Best Response (IBR) part of the algorithm                      %
%                                                                         %
% Inputs:                                                                 %
% -cdf:             the CDF of values for each node [N x V]               %
% -psi:             the transmission attempt cost [scalar, R+]            %
% -epsilon:         error threshold to stop IBR [scalar, 0-1]             %
% -values:          the possible values for all nodes [1 x V]             %
% -init_thresholds: the initial thresholds to start IBR [1 x N]           %
% -max_iter:        the maximum number of IBR iterations [scalar, int]    %
%                                                                         %
% Outputs:                                                                %
% -thresholds:      the final thresholds at convergence [1 x N]           %
% -reward_history:  the reward for each IBR iteration [1 x max_iter]      %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Utility variables
N = length(init_thresholds);
V = size(values, 2);
nodes = 1 : N;
reward_history = zeros(1, N * max_iter + 1);

% Pre-compute transmission values
tx_values = zeros(size(cdf));
for n = 1 : N
    tx_values(n, 1) = diff(cdf(n, :)) * values';
    for v = 2 : V
        tx_values(n, v) = tx_values(n, v - 1) - values(v) * (cdf(n, v) - cdf(n, v - 1));
    end
end


previous_thresholds = -ones(1, N);
thresholds = init_thresholds;
iter = 1;
% Compute initial value
for n = 1 : N
    threshold_index = find(cdf(n, :) >= thresholds(n), 1);
    if (~isempty(threshold_index))
        reward_history(1) = reward_history(1) + prod(thresholds(1 : end ~= n)) * tx_values(n, threshold_index) - psi * (1 - thresholds(n));
    end
end

% Iterated best response
while(iter < max_iter && max(abs(previous_thresholds - thresholds)) >= epsilon)
    previous_thresholds = thresholds;
    % Iterate over the nodes
    for n = nodes
        % Find best response
        theta = 0;
        zeta = prod(thresholds(1 : end ~= n));
        for m = nodes(1 : end ~= n)
            threshold_index = find(cdf(m, :) >= thresholds(m), 1);
            if (~isempty(threshold_index))
                theta = theta + tx_values(m, threshold_index) ./ thresholds(m);            
            end
        end
        theta = theta + psi / zeta;
        value_index = find(values >= theta, 1);
        if (isempty(value_index))
            thresholds(n) = 1;
        else
            thresholds(n) = cdf(n, value_index);
        end
        % Compute value
        for m = nodes
            threshold_index = find(cdf(m, :) >= thresholds(m), 1);
            if (~isempty(threshold_index))
                reward_history(N * (iter - 1) + n + 1) = reward_history(N * (iter - 1) + n + 1) + prod(thresholds(1 : end ~= m)) * tx_values(m, threshold_index) - psi * (1 - thresholds(m));
            end
        end
    end
    iter = iter + 1;
end
reward_history(iter : end) = max(reward_history);