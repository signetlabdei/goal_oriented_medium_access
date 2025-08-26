
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                         %
%                          script: sensors_symm                           %
%           author: Federico Chiariotti (chiariot@dei.unipd.it)           %
%                             license: GPLv3                              %
%                                                                         %
%                                                                         %
%                                                                         %
% Runs a Monte Carlo simulation with a symmetric sensor network using     %
% chi-squared value distributions                                         %
%                                                                         %
% Parameters:                                                             %
% -Ns:          the maximum number of nodes [scalar]                      %
% -M:           the number of steps to simulate [scalar]                  %
% -epsilon:     the NE approximation error [scalar]                       %
% -delta:       the VoI quantization step [scalar]                        %
% -Vmax:        the maximum possible VoI [scalar]                         %
% -psi:         the transmission attempt cost [scalar]                    %
% -max_iter:    the maximum number of IBR iterations [scalar]             %
% -K:           number of cleared slots in BT [scalar]                    %
% -p1:          alpha for ZW/GZW/LZW [scalar]                             %
% -p2:          beta for GZW/LZW [scalar]                                 %
% -M:           the maximum AoII [scalar]                                 %
% -omegas:      the feedback error probabilities                          %
% -setting:     0 for simulating over sigma, 1 for simulating over nu     %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

close all
clearvars

% Simulation parameters
Ns = 100;
M = 1e6;
epsilon = 1e-3;
delta = 1e-4;
psi = 0;
max_iter = 1000;
Vmax = 20;

% Chi-squared distribution
values = 0 : delta : Vmax;
cdf = [0, 1 - exp(-(values + delta / 2))];

% Compute pull-based solution
exp_values = diff(cdf) * values';
pull_reward = 0;
pull_rho = 0;
% Iterate over values
for v = 1 : length(values)
    % Compute expected value
    if (v > 1)
        exp_values = exp_values - values(v) * (cdf(v) - cdf(v - 1));
    end
    v_reward = exp_values - psi * (1 - cdf(v));
    % Check if reward improves
    if (v_reward > pull_reward)
        pull_rho = 1 - cdf(v);
        pull_reward = v_reward;
    end
end

% Compute pull-based performance
pull_rewards = ones(1, Ns) * pull_reward;
pull_channel_uses = ones(1, Ns) * pull_rho;
pull_energies = ones(1, Ns) * pull_rho;
pull_goodputs = pull_channel_uses;

% Auxiliary variables for LIBRA
th_rewards = zeros(1, Ns);
th_energies = zeros(1, Ns);
th_thresholds = zeros(Ns, Ns);
mc_rewards = zeros(1, Ns);
mc_energies = zeros(1, Ns);
mc_channel_uses = zeros(1, Ns);
mc_goodputs = zeros(1, Ns);

% Run LIBRA
for N = 2 : Ns
    N
    cdfs = repmat(cdf, N, 1);
    % Determine solution
    [v_eq, voi_0, initial_thresholds] = equal_value_initialization(cdfs, values, psi);
    [thresholds, reward_iter] = iterated_best_response(cdfs, psi, epsilon, values, initial_thresholds, max_iter);
    th_rewards(N) = max(reward_iter);
    th_energies(N) = sum(1 - thresholds);
    th_thresholds(N, 1 : N) = thresholds(1);
    % Monte Carlo check
    [mc_voi, mc_reward, energy, goodput, channel_use, reward_history] = montecarlo(cdfs, values, psi, thresholds, M);
    mc_rewards(N) = mc_reward;
    mc_energies(N) = sum(energy);
    mc_channel_uses(N) = channel_use;
    mc_goodputs(N) = goodput;
end