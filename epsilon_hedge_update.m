function [new_exp_values] = epsilon_hedge_update(exp_values, thetas, gamma, psi, vois, actions, outcome, alphas, betas, lambdas, rhos)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                         %
%                     function: epsilon_hedge_update                      %
%           author: Federico Chiariotti (chiariot@dei.unipd.it)           %
%                             license: GPLv3                              %
%                                                                         %
%                                                                         %
%                                                                         %
% Computes a MAMAB update by using counterfactual reasoning               %
%                                                                         %
% Inputs:                                                                 %
% -exp_values:      the current arm rewards [N x V]                       %
% -thetas:          the possible BETA thresholds [1 x V]                  %
% -gamma:           the learning rate for this step [scalar, 0-1]         %
% -psi:             the transmission attempt cost [scalar, R+]            %
% -vois:            the VoI for this step for all nodes [N x 1]           %
% -actions:         whether each node transmitted [1 x N, bool]           %
% -outcome:         the ID of the successful node, 0 for silence, or -1   %
%                   for a collision [scalar, int]                         %
% -alphas:          ratio of occupied slots if node n is silent [1 x N]   %
% -betas:           ratio of successful slots if node n is silent [1 x N] %
% -lambdas:         average VoI if node n is silent [1 x N]               %
% -rhos:            estimated transmission rates [1 x N]                  %
%                                                                         %
% Outputs:                                                                %
% -new_exp_values:  the updated arm rewards [N x V]                       %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Utility variables
N = size(exp_values, 1);
V = size(exp_values, 2);
new_exp_values = zeros(N, V);

% Outcome: silence
if (outcome == 0)
    for n = 1 : N
        tx_idx = find(thetas > vois(n), 1);
        if (~isempty(tx_idx))
            % The node transmits: success
            new_exp_values(n, 1 : tx_idx - 1) = exp_values(n, 1 : tx_idx - 1) + gamma * (vois(n) - psi);
            % The node is silent: silence
            new_exp_values(n, tx_idx : end) = exp_values(n, tx_idx : end);
        else
            new_exp_values(n, :) = exp_values(n, :) + gamma * (vois(n) - psi);
        end
    end
end


% Outcome: success
if (outcome > 0)
    % Successful node
    tx_idx = find(thetas > vois(outcome), 1);
    if (~isempty(tx_idx))
        % The node transmits: success
        new_exp_values(outcome, 1 : tx_idx - 1) = exp_values(outcome, 1 : tx_idx - 1) + gamma * (vois(outcome) - psi);
        % The node is silent: silence
        new_exp_values(outcome, tx_idx : end) = exp_values(outcome, tx_idx : end);
    else
        new_exp_values(outcome, :) = exp_values(outcome, :) + gamma * (vois(outcome) - psi);
    end
    % Other nodes
    for n = setdiff(1 : N, outcome)
        tx_idx = find(thetas > vois(n), 1);
        if (~isempty(tx_idx))
            % The node transmits: collision
            new_exp_values(n, 1 : tx_idx - 1) = exp_values(n, 1 : tx_idx - 1) - gamma * 2 * psi;
            % The node is silent: success (same VoI as the real outcome)
            new_exp_values(n, tx_idx : end) = exp_values(n, tx_idx : end) + gamma * (vois(outcome) - psi);
        else
            new_exp_values(n, :) = exp_values(n, :) - gamma * 2 * psi;
        end
    end
end

% Outcome: collision
if (outcome == -1)
    for n = setdiff(1 : N, outcome)
        if (actions(n) == 0)
            % The node was not a part of the collision set: collision is
            % unavoidable
            tx_idx = find(thetas > vois(n), 1);
            activity = sum(rhos) - rhos(n) - betas(n);
            collisions = alphas(n) - betas(n);
            if (collisions > 0)
                if (~isempty(tx_idx))
                    % The node transmits: collision (involving the node!)
                    new_exp_values(n, 1 : tx_idx - 1) = exp_values(n, 1 : tx_idx - 1) - gamma * psi * (activity + 1) / collisions;
                    % The node is silent: collision
                    new_exp_values(n, tx_idx : end) = exp_values(n, tx_idx : end) - gamma * psi * activity / collisions;
                else
                    new_exp_values(n, :) = exp_values(n, :) - gamma * psi * (activity + 1) / collisions;
                end
            else
                new_exp_values(n, :) = exp_values(n, :);
            end
        else
            % The node was a part of the collision set: collision might be
            % avoidable if it is silent
            tx_idx = find(thetas > vois(n), 1);
            activity = sum(rhos) - rhos(n);
            if (~isempty(tx_idx))
                % The node transmits: collision (involving the node!)
                new_exp_values(n, 1 : tx_idx - 1) = exp_values(n, 1 : tx_idx - 1) - gamma * psi * (activity + 1) / alphas(n);
                % The node is silent: collision or success
                new_exp_values(n, tx_idx : end) = exp_values(n, tx_idx : end) +  gamma * (lambdas(n) - psi * activity / alphas(n));
            else
                new_exp_values(n, :) = exp_values(n, :) - gamma * psi * (activity + 1) / alphas(n);
            end
        end
    end
end

end