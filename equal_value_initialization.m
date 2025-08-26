function [v_eq, reward, thresholds] = equal_value_initialization(cdf, values, psi)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                         %
%                  function: equal_value_initialization                   %
%           author: Federico Chiariotti (chiariot@dei.unipd.it)           %
%                             license: GPLv3                              %
%                                                                         %
%                                                                         %
%                                                                         %
% Determines the VoI quantile thresholds considering the same actual VoI  %
% as the threshold for all nodes                                          %
%                                                                         %
% Inputs:                                                                 %
% -cdf:             the CDF of values for each node [N x V]               %
% -values:          the possible values for all nodes [1 x V]             %
% -psi:             the transmission attempt cost [scalar, R+]            %
%                                                                         %
% Outputs:                                                                %
% -v_eq:            the value of the transmission threshold [scalar]      %
% -reward:          the reward of the initialization policy [scalar]      %
% -thresholds:      the threshold quantiles [1 x N]                       %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Utility variables
N = size(cdf, 1);
V = length(values);
% Outputs
v_eq = -1;
reward = 0;
thresholds = zeros(1, N);

tx_quantile = zeros(1, N);
exp_values = zeros(1, N);

% Expected value from transmission (distribution mean)
for n = 1 : N
    exp_values(n) = diff(cdf(n, :)) * values';
end

% Iterate over possible values
for v = 1 : V - 1
    % Compute values
    for n = 1 : N
        tx_quantile(n) = cdf(n, v);
        if (v > 1)
            exp_values(n) = exp_values(n) - values(v) * (cdf(n, v) - cdf(n, v - 1));
        end
    end
    value = 0;
    % Compute value
    for n = 1 : N
        value = value + prod(tx_quantile(1 : end ~= n)) * exp_values(n) - psi * (1 - tx_quantile(n));
    end
    % Update best value
    if (value > reward)
        reward = value;
        v_eq = values(v);
        thresholds = tx_quantile;
    end
end