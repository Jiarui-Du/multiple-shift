% RUN_KOROBOV_APPROXIMATION Evaluates the function approximation convergence.
%
% Purpose:
%   Reproduces the numerical approximation results (Figures 3, 4, and 5) from 
%   the paper. It evaluates the L_2 or L_\infty approximation error of the 
%   deterministic multiple-shift lattice algorithm on specific test functions 
%   (f_1 or f_2) across varying dimensions and base lattice sizes N.
%   It strictly verifies that the optimal convergence rate O(N_{tot}^{-\alpha}) 
%   is preserved despite the deterministic constraints.

clear all;
close all;
format compact;
format short;
clc;
addpath("./functions")
rng(2026); % Fix random seed for reproducibility

%% 1. Experimental Parameters Setup
dim_list = [2,4];
num_dims = length(dim_list);

N_list_pool = 2.^(8:18); 
num_N = length(N_list_pool);
for i = 1:num_N
    N0 = N_list_pool(i);
    N_list(i) = nearprime(N0);
end

% Algorithm & Function Parameters
alpha = 2.5;                % Smoothness parameter (2.5 for f_1, 1.5 for f_2)
error_metric = inf;           % Metric selector: 2 for L_2, inf for L_\infty
threshold_t = 0.95;         % Tolerance controlling the condition number bound
num_mc_shifts = 10;         % Number of random shifts for out-of-sample error evaluation
num_eval_pts = 500000;      % Number of out-of-sample points for numerical integration

% Target function f_1 definition
target_func = @(x) prod((x - 1/2).^2 .* sin(2 * pi * x - pi), 2);
% Target function f_2 definition
% target_func = @(x) prod(121*sqrt(33)/100*max(25/121-(x-(1/2)).^2,0), 2);
norm_f = 1; % 1 for Absolute Error Or theoretical norm of the target function for Relative Error

%% 2. Parallel Task Preparation (Flattening for parfor)
num_tasks = num_dims * num_N;
[task_n_idx_grid, task_d_idx_grid] = meshgrid(1:num_N, 1:num_dims);
task_n_indices = task_n_idx_grid(:);
task_d_indices = task_d_idx_grid(:);

% Pre-allocate 1D arrays for parfor compatibility
res_errors  = zeros(num_tasks, 1);
res_S_vals  = zeros(num_tasks, 1);
res_nA      = zeros(num_tasks, 1);
res_max_len = zeros(num_tasks, 1);

%% 3. Main Parallel Computation Loop
% parpool(2)
tic;
for k = 1:num_tasks
    n_idx = task_n_indices(k);
    d_idx = task_d_indices(k);
    
    N = N_list(n_idx);
    d = dim_list(d_idx);
    
    % Dimension-dependent weights \gamma_j
    gamma = 2.^((1 - (1:d)) / 10);
    
    % Generate strictly randomized out-of-sample evaluation points via Sobol sequence
    sobol_seq = scramble(sobolset(d), 'MatousekAffineOwen');
    eval_pts = net(sobol_seq, num_eval_pts);
    
    % Generate random shifts for robust MC error estimation
    mc_shifts = rand(d, num_mc_shifts);
    
    % Compute optimal truncation bound M
    M = find_M_optimal(N, d, alpha, gamma);
    
    % Execute the fully deterministic multiple-shift approximation
    [err_val, S_val, nA, max_len] = evaluate_approximation_error( ...
        N, M, d, alpha, gamma, threshold_t, num_mc_shifts, mc_shifts, ...
        target_func, norm_f, eval_pts, error_metric);
        
    fprintf('N = %6d, d = %d | S = %2d, |A(M)| = %6d, R = %1d, Error = %.3e\n', ...
        N, d, S_val, nA, max_len, err_val);
        
    res_errors(k)  = err_val;
    res_S_vals(k)  = S_val;
    res_nA(k)      = nA;
    res_max_len(k) = max_len;
end
total_time = toc;
fprintf('Total parfor loop time: %.2f seconds\n', total_time);

%% 4. Data Restructuring & Archiving
error_matrix  = reshape(res_errors, num_dims, num_N);
s_matrix      = reshape(res_S_vals, num_dims, num_N);
nA_matrix     = reshape(res_nA, num_dims, num_N);
max_len_matrix= reshape(res_max_len, num_dims, num_N);

if error_metric == 2
    metric_str = 'L2';
else
    metric_str = 'Linf';
end
alpha_str = sprintf('alpha%d', floor(alpha));
func_label = sprintf('$f_%d$', floor(alpha) - 1); % Maps alpha=2.5 to f_1, 1.5 to f_2

%% Save result
if ~exist('results', 'dir'), mkdir('results'); end
filename = sprintf('results/korobov_approx_%s_%s.mat', alpha_str, metric_str);
save(filename, 'dim_list', 'alpha', 'N_list', 'error_matrix', 's_matrix', 'nA_matrix', 'max_len_matrix');

%% Load result
% if error_metric == 2
%     metric_str = 'L2';
% else
%     metric_str = 'Linf';
% end
% alpha_str = sprintf('alpha%d', floor(alpha));
% func_label = sprintf('$f_%d$', floor(alpha) - 1);
% filename = sprintf('results/korobov_approx_%s_%s.mat', alpha_str, metric_str);
% load(filename)

%% 5. Visualization 
if ~exist('figs', 'dir'), mkdir('figs'); end

% Extract data specifically for dimensions d=2 and d=4 as shown in the paper
target_dims = [2, 4];
target_indices = find(ismember(dim_list, target_dims));
num_target_dims = length(target_dims);

filtered_dim_list = dim_list(target_indices);
filtered_errors = error_matrix(target_indices, :);
filtered_s = s_matrix(target_indices, :);

% --- Plot 1: Convergence wrt Base Lattice Size N ---
figure('Position', [100, 100, 600, 500]);
plot_lattice_error(alpha, num_target_dims, filtered_dim_list, N_list, filtered_errors, [], error_metric, 9);
title(sprintf('Convergence for %s ($\\alpha = %.1f$)', func_label, alpha), 'Interpreter', 'latex', 'FontSize', 16);
save_figure_eps(gcf, sprintf('figs/korobov_approx_%s_%s_N.eps', alpha_str, metric_str));

% --- Plot 2: Convergence wrt Total Cost (N * S) ---
figure('Position', [150, 150, 600, 500]);
plot_total_error(alpha, num_target_dims, filtered_dim_list, filtered_s, N_list, filtered_errors, [], error_metric, 9);
title(sprintf('Convergence for %s ($\\alpha = %.1f$)', func_label, alpha), 'Interpreter', 'latex', 'FontSize', 16);
save_figure_eps(gcf, sprintf('figs/korobov_approx_%s_%s_tot.eps', alpha_str, metric_str));

% --- Plot 3: Quadratic Scaling of Shifts S wrt log2(N) ---
log2_N = log2(N_list);
S_d2 = s_matrix(find(dim_list == 2), :);
S_d4 = s_matrix(find(dim_list == 4), :);

% Perform quadratic polynomial fitting
p_d2 = polyfit(log2_N, S_d2, 2);
p_d4 = polyfit(log2_N, S_d4, 2);

x_fit = linspace(min(log2_N), max(log2_N), 100);
S_fit_d2 = polyval(p_d2, x_fit);
S_fit_d4 = polyval(p_d4, x_fit);

figure('Position', [200, 200, 650, 450]); 
hold on; grid on; box on;

h1 = plot(log2_N, S_d2, 'bs', 'MarkerSize', 10, 'MarkerFaceColor', 'b');
h2 = plot(log2_N, S_d4, 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');

plot(x_fit, S_fit_d2, 'b--', 'LineWidth', 2);
plot(x_fit, S_fit_d4, 'r--', 'LineWidth', 2);

set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', 12);
xlabel('$\log_2(N)$', 'Interpreter', 'latex', 'FontSize', 14);
ylabel('Number of shifts $S$', 'Interpreter', 'latex', 'FontSize', 14);

% Construct formatted legend strings for the quadratic models
leg_str_d2 = sprintf('$d=2$ Fit: $S \\approx %.2f (\\log_2(N))^2 %+.2f (\\log_2(N)) %+.2f$', p_d2(1), p_d2(2), p_d2(3));
leg_str_d4 = sprintf('$d=4$ Fit: $S \\approx %.2f (\\log_2(N))^2 %+.2f (\\log_2(N)) %+.2f$', p_d4(1), p_d4(2), p_d4(3));

legend([h1, h2], {leg_str_d2, leg_str_d4}, 'Interpreter', 'latex', 'Location', 'northwest', 'FontSize', 11);
title(sprintf('Shift Scaling for %s ($\\alpha = %.1f$)', func_label, alpha), 'Interpreter', 'latex', 'FontSize', 16);

fprintf('\n--- Quadratic Fit Coefficients (S = a*x^2 + b*x + c) ---\n');
fprintf('d=2: a = %7.4f, b = %7.4f, c = %7.4f\n', p_d2(1), p_d2(2), p_d2(3));
fprintf('d=4: a = %7.4f, b = %7.4f, c = %7.4f\n', p_d4(1), p_d4(2), p_d4(3));

save_figure_eps(gcf, sprintf('figs/korobov_approx_%s_%s_S_N.eps', alpha_str, metric_str));

% =========================================================================
% Local Helper Functions
% =========================================================================
function save_figure_eps(fig_handle, file_path)
    set(fig_handle, 'Color', 'w');
    set(fig_handle, 'PaperPositionMode', 'auto');
    print(fig_handle, file_path, '-depsc', '-vector');
end