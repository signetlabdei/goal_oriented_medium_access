function [actions, greedy] = epsilon_hedge(thetas, exp_values, epsilon)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                         %
%                         function: epsilon_hedge                         %
%           author: Federico Chiariotti (chiariot@dei.unipd.it)           %
%                             license: GPLv3                              %
%                                                                         %
%                                                                         %
%                                                                         %
% Run the epsilon-hedge MAMAB algorithm                                   %
%                                                                         %
% Inputs:                                                                 %
% -thetas:          the possible BETA thresholds [1 x V]                  %
% -exp_values:      the current arm rewards [N x V]                       %
% -epsilon:         exploration rate of epsilon-hedge [scalar, 0-1]       %
%                                                                         %
% Outputs:                                                                %
% -actions:         the action for each node [1 x N]                      %
% -greedy:          the highest-reward option for each node [1 x N]       %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Utility variables
N = size(exp_values, 1);
V = size(exp_values, 2);

actions = zeros(1, N);
greedy = zeros(1, N);

% Select actions
for n = 1 : N
    % Softmax selection
    values = exp_values(n, :)';
    [~, greedy(n)] = max(values);
    if (rand < epsilon)
        % Exploration: do not transmit
        prob = ones(1, V) / V;
    else
        % Softmax selection
        prob = softmax(values);
    end
    actions(n) = datasample(thetas, 1, 'Weights', prob);
end