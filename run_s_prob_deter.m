% RUN_S_PROB_DETER Evaluates sampling complexity and numerical stability.
%
% Purpose:
%   Reproduces Figure 2 from the paper. This script compares the Adaptive
%   Deterministic Construction (Algorithm 3) against state-of-the-art
%   probabilistic baseline methods in terms of total sampling cost (S) and
%   the numerical stability (Maximum Condition Number of the Gram matrix G).
%
% Methods Compared:
%   1. Probabilistic (99% Success): Requires S = O(R^2 \log N) independent shifts.
%   2. Simplified Probabilistic (99%): Uses the same uniform shift for the fiber.
%   3. Adaptive Deterministic: Our proposed Algorithm 3 guaranteeing cond(G) <= (1+t)/(1-t).

clear all;
close all;
format short;
clc;
addpath("./functions")
rng(26); % Fix random seed for strict reproducibility

%% 1. Parameters Setup
N_list_pool = 2.^(10:20); 
num_N = length(N_list_pool);
for i = 1:num_N
    N0 = N_list_pool(i);
    N_list(i) = nearprime(N0);
end
dim_list = [2:5, 10, 15, 20, 25, 30, 50];
num_dims = length(dim_list);

alpha = 1;         % Smoothness parameter
threshold_t = 0.95; % Error tolerance controlling the condition number bound
num_tasks = num_N * num_dims;

% Pre-generate task list by flattening the loops for parfor efficiency
[task_n_idx_grid, task_d_idx_grid] = meshgrid(1:num_N, 1:num_dims);
task_n_indices = task_n_idx_grid(:);
task_d_indices = task_d_idx_grid(:);

%% 2. Pre-allocate Output Arrays (1D arrays to support parfor)
res_S_prob       = zeros(num_tasks, 1);
res_S_simp_prob  = zeros(num_tasks, 1);
res_S_adap   = zeros(num_tasks, 1);

res_cond_prob      = zeros(num_tasks, 1);
res_cond_simp_prob = zeros(num_tasks, 1);
res_cond_adap  = zeros(num_tasks, 1);

res_R      = zeros(num_tasks, 1);
res_N_real = zeros(num_tasks, 1);
res_nA     = zeros(num_tasks, 1);

%% 3. Main Computation (Parallel Execution)
tic;
parfor k = 1:num_tasks
    % Parse parameters for the current task
    d_idx = task_d_indices(k);
    n_idx = task_n_indices(k);

    d = dim_list(d_idx);
    N = N_list(n_idx);
    gamma = 2.^((1 - (1:d)) / 10);
    % gamma = 0.9.^((1:d)-1);

    % Construct generating vector and lattice fibers
    g = construct_generating_vector_cbc(N, d, gamma, alpha);
    M = find_M_optimal(N, d, alpha, gamma);
    [A, fibers, ~] = construct_lattice_fibers(N, M, g, gamma, alpha);

    pmin = floor((M*gamma(1))^(1/alpha))*floor((M*gamma(2))^(1/alpha))/N;    

    fiber_lengths = cellfun(@length, fibers);
    R = max(fiber_lengths);
    nA = length(A);

    res_nA(k) = nA;
    res_R(k) = R;
    res_N_real(k) = N;

    % --- Probabilistic Methods Bounds ---
    % Constant K ensuring 99% success probability based on theoretical bounds
    K_prob = 1 - log(0.01 / R^2) / log(N);
    S_trials_prob = ceil(2 * K_prob * R * log(N));
    S_prob = S_trials_prob * R; % Traditional method samples R points per trial

    K_simp = 1 - log(0.01) / log(N);
    S_simp_prob = ceil(2 * K_simp * R * log(N) / threshold_t^2);

    % Generate random uniform shifts for probabilistic baselines
    Y_prob = rand(R, d, S_trials_prob);
    Y_simp_prob = rand(S_simp_prob, d);

    % --- Adaptive Deterministic Construction ---
    [Y_adap, S_adap, ~] = adaptive_construction_new(d, A, fibers, R, threshold_t, pmin);
    
    % Arrays to store the condition numbers for all N fibers
    cond_G_prob      = zeros(N, 1);
    cond_G_simp_prob = zeros(N, 1);
    cond_G_adap  = zeros(N, 1);

    % Evaluate numerical stability (Condition Number) over all fibers
    for r = 0:N-1
        idx = fibers{r+1};
        v = length(idx);
        if v > 0
            Gamma = A(idx, :); % Dimensions: v x d

            % 1. Evaluate Gram Matrix for Traditional Probabilistic Method
            % Vectorized reconstruction of the shift matrix across all trials
            Y_sub = Y_prob(1:v, :, :);
            Y_stack = reshape(permute(Y_sub, [1, 3, 2]), v * S_trials_prob, d);
            B_prob = exp(2 * pi * 1i * (Y_stack * Gamma.'));
            G_prob = B_prob' * B_prob;
            cond_G_prob(r+1) = cond(G_prob);

            % 2. Evaluate Gram Matrix for Simplified Probabilistic Method
            B_simp = exp(2 * pi * 1i * (Y_simp_prob * Gamma.'));
            G_simp = B_simp' * B_simp;
            cond_G_simp_prob(r+1) = cond(G_simp);

            % 3. Evaluate Gram Matrix for Adaptive Method (Random)
            B_adap = exp(2 * pi * 1i * (Y_adap * Gamma.'));
            G_adap = B_adap' * B_adap;
            cond_G_adap(r+1) = cond(G_adap);
        end
    end

    % Record the worst-case (maximum) condition number across all fibers
    res_cond_prob(k)      = max(cond_G_prob);
    res_cond_simp_prob(k) = max(cond_G_simp_prob);
    res_cond_adap(k)  = max(cond_G_adap);

    res_S_prob(k)      = S_prob;
    res_S_simp_prob(k) = S_simp_prob;
    res_S_adap(k)  = S_adap;
    fprintf('Completed N = %d, d = %d\n', N, d);
end

total_time = toc;
fprintf('Total computation time: %.2f seconds\n', total_time);

%% 4. Data Restructuring & Saving
Data.S_prob       = reshape(res_S_prob, num_dims, num_N);
Data.S_simp_prob  = reshape(res_S_simp_prob, num_dims, num_N);
Data.S_adap   = reshape(res_S_adap, num_dims, num_N);

Data.cond_prob      = reshape(res_cond_prob, num_dims, num_N);
Data.cond_simp_prob = reshape(res_cond_simp_prob, num_dims, num_N);
Data.cond_adap      = reshape(res_cond_adap, num_dims, num_N);

Data.R      = reshape(res_R, num_dims, num_N);
Data.N_real = reshape(res_N_real, num_dims, num_N);
Data.nA     = reshape(res_nA, num_dims, num_N);

%% Save result
if ~exist('results', 'dir'), mkdir('results'); end
save('results/S_eff.mat', 'Data', 'dim_list', 'N_list', 'num_N');

%% Load result
% load('results/S_eff.mat')

%% 5. Visualization (Reproducing Figure 2)
% High-contrast academic color palette
colors = lines(4);
c_prob = colors(2,:);  % Orange
c_simp = colors(1,:);  % Blue
c_adap = colors(3,:);
% c_adap = [0.1, 0.7, 0.2]; % Green

lw = 1.5;
ms = 6;

idx_d_high = find(dim_list == 50);
idx_N_max = num_N; 
N_exponents = log2(N_list);
IN_exponents = round(N_exponents);

figure('Position', [100, 100, 1200, 350]);

% --- Panel 1: Scalability wrt Lattice Size N (Fixed d = 30) ---
subplot(1, 3, 1);
hold on; grid on; box on;
semilogy(N_exponents, Data.S_prob(idx_d_high,:), '-s', 'Color', c_prob, 'LineWidth', lw, 'MarkerSize', ms);
semilogy(N_exponents, Data.S_simp_prob(idx_d_high,:), '-d', 'Color', c_simp, 'LineWidth', lw, 'MarkerSize', ms);
semilogy(N_exponents, Data.S_adap(idx_d_high,:), '-o',  'Color', c_adap, 'LineWidth', 2, 'MarkerFaceColor', c_adap);
% semilogy(N_exponents, N_exponents.^2 / 2, '-.^',  'Color', [0.5, 0.5, 0.5], 'LineWidth', 2); % Reference curve
xlim([N_exponents(1), N_exponents(end)*1.05]);
ylim([0, max(Data.S_prob(idx_d_high,:)) * 1.3]);
ax = gca; ax.YColor = 'k';

xtick_labels = arrayfun(@(x) sprintf('$2^{%d}$', x), IN_exponents, 'UniformOutput', false);
set(gca, 'XTick', N_exponents, 'TickLabelInterpreter', 'latex', 'XTickLabel', xtick_labels);
xlabel('Base Lattice Size $N$', 'Interpreter', 'latex');
ylabel('The Total Number of Shifts $S$', 'Interpreter', 'latex');
title('$d=50$', 'Interpreter', 'latex');

leg = legend('Probabilistic', 'Simplified Probabilistic','Adaptive', ...
    'Location', 'northwest', 'Interpreter', 'latex');
set(leg, 'FontSize', 9);

yyaxis right
ax.YColor = [0, 0, 0];
plot(N_exponents, Data.R(idx_d_high, :), '--s', 'Color', [0.4660, 0.6740, 0.1880], ...
    'LineWidth', 1.5, 'MarkerSize', 8, 'MarkerFaceColor', 'w','DisplayName', 'Max Fiber Length $R$');
% ylabel('Max Fiber Length $R$', 'Interpreter', 'latex');
ylim([0, max(Data.R(idx_d_high, :))+3]);

% --- Panel 2: Total Shifts S vs Dimension d (Fixed N = 2^16) ---
subplot(1, 3, 2);
hold on; grid on; box on;
semilogy(dim_list, Data.S_prob(:,idx_N_max), '-s', 'Color', c_prob, 'LineWidth', lw, 'MarkerSize', ms);
semilogy(dim_list, Data.S_simp_prob(:,idx_N_max), '-d', 'Color', c_simp, 'LineWidth', lw, 'MarkerSize', ms);
semilogy(dim_list, Data.S_adap(:,idx_N_max), '-o',  'Color', c_adap, 'LineWidth', 2, 'MarkerFaceColor', c_adap);

xlim([dim_list(1), dim_list(end)*1.05]);
ax = gca;
set(gca, 'TickLabelInterpreter', 'latex');
xlabel('Dimension $d$', 'Interpreter', 'latex');
% ylabel('Number of shifts $S$', 'Interpreter', 'latex');
title(sprintf('$N = %d$', N_list(idx_N_max)), 'Interpreter', 'latex');

yyaxis right
ax.YColor = [0, 0, 0];
plot(dim_list, Data.R(:,idx_N_max), '--s', 'Color', [0.4660, 0.6740, 0.1880], ...
    'LineWidth', 1.5, 'MarkerSize', 8, 'MarkerFaceColor', 'w');
ylabel('Max Fiber Length $R$', 'Interpreter', 'latex');
ylim([0, max(Data.R(:,idx_N_max))+1]);

% --- Panel 3: Maximum Condition Number vs Dimension d (Fixed N = 2^16) ---
subplot(1, 3, 3);
hold on; grid on; box on;
semilogy(dim_list, Data.cond_prob(:,idx_N_max), '-s', 'Color', c_prob, 'LineWidth', lw, 'MarkerSize', ms);
semilogy(dim_list, Data.cond_simp_prob(:,idx_N_max), '-d', 'Color', c_simp, 'LineWidth', lw, 'MarkerSize', ms);
semilogy(dim_list, Data.cond_adap(:,idx_N_max), '-o',  'Color', c_adap, 'LineWidth', 2, 'MarkerFaceColor', c_adap);

xlim([dim_list(1), dim_list(end)*1.05]);
ax = gca;
set(gca, 'TickLabelInterpreter', 'latex');
xlabel('Dimension $d$', 'Interpreter', 'latex');
ylabel('Max condition number of $G$', 'Interpreter', 'latex');
ylim([0.9, max(Data.cond_adap(:, idx_N_max)) * 1.2]);
title(sprintf('$N = %d$', N_list(idx_N_max)), 'Interpreter', 'latex');

yyaxis right
ax.YColor = [0, 0, 0];
plot(dim_list, Data.R(:,idx_N_max), '--s', 'Color', [0.4660, 0.6740, 0.1880], ...
    'LineWidth', 1.5, 'MarkerSize', 8, 'MarkerFaceColor', 'w');
ylabel('Max Fiber Length $R$', 'Interpreter', 'latex');
ylim([0, max(Data.R(:,idx_N_max))+1]);

set(findall(gcf,'-property','FontSize'),'FontSize', 15);
if ~exist('figs', 'dir'), mkdir('figs'); end
print(gcf, 'figs/numerical_stability_comparison.eps', '-depsc', '-vector');