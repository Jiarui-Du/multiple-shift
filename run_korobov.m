% RUN_KOROBOV_APPROXIMATION Evaluates the function approximation convergence.
%
% Purpose:
%   Reproduces the numerical approximation results (Figures 3, 4, and 5) from 
%   the paper. It evaluates the L_2 or L_\infty approximation error of the 
%   deterministic multiple-shift lattice algorithm on specific test functions 
%   (f_1 or f_2) across varying dimensions and base lattice sizes N.
%   Errors are estimated on a separate set of test points.

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

N_list_pool = 2.^(8:20);
num_N = length(N_list_pool);
for i = 1:num_N
    N0 = N_list_pool(i);
    N_list(i) = nearprime(N0);
end

% Algorithm & Function Parameters
alpha = 2.5;                % Smoothness parameter (2.5 for f_1, 1.5 for f_2)
error_metric = 2;           % Metric selector: 2 for L_2, inf for L_\infty
threshold_t = 0.95;         % Tolerance controlling the condition number bound
num_mc_shifts = 10;         % Number of random shifts for out-of-sample error evaluation
num_eval_pts = 2^18;        % Number of out-of-sample points for numerical integration

% Target function f_1 definition
target_func = @(x) prod((x - 1/2).^2 .* sin(2 * pi * x - pi), 2);
% Target function f_2 definition
% target_func = @(x) prod(121*sqrt(33)/100*max(25/121-(x-(1/2)).^2,0), 2);
norm_f = 1; % 1 for Absolute Error Or theoretical norm of the target function for Relative Error

%% 2. Task Preparation
num_tasks = num_dims * num_N;
[task_n_idx_grid, task_d_idx_grid] = meshgrid(1:num_N, 1:num_dims);
task_n_indices = task_n_idx_grid(:);
task_d_indices = task_d_idx_grid(:);

% Pre-allocate results in task order
res_errors  = zeros(num_tasks, 1);
res_S_vals  = zeros(num_tasks, 1);
res_nA      = zeros(num_tasks, 1);
res_max_len = zeros(num_tasks, 1);
res_frequency_time = zeros(num_tasks, 1);
res_setup_time = zeros(num_tasks, 1);
res_recovery_time = zeros(num_tasks, 1);
res_evaluation_time = zeros(num_tasks, 1);

%% 3. Main Serial Computation Loop
total_clock = tic;
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
    
    % Section 6 takes the frequency set as input.
    frequency_clock = tic;
    M = find_M_optimal(N, d, alpha, gamma);
    A = construct_frequency_set(N, M, d, gamma, alpha);
    res_frequency_time(k) = toc(frequency_clock);

    setup_clock = tic;
    g = construct_generating_vector_cbc(N, d, gamma, alpha);
    [fibers, ~] = construct_lattice_fibers(A, g, N);
    nA = size(A, 1);
    max_len = max(cellfun(@length, fibers));
    if d > 1
        pmin = floor((M*gamma(1))^(1/alpha))*floor((M*gamma(2))^(1/alpha))/N;
    else
        pmin = max_len;
    end
    [Y, S_val, ~] = adaptive_construction_new(d, A, fibers, max_len, threshold_t, pmin);

    prepared = prepare_lattice_recovery(A, fibers, Y);
    res_setup_time(k) = toc(setup_clock);
    clear fibers A
    
    % Sample and recover using the same precomputation for every global shift.
    [err_val, recovery_time, evaluation_time] = evaluate_approximation_error( ...
        g, Y, prepared, num_mc_shifts, mc_shifts, ...
        target_func, norm_f, eval_pts, error_metric);
    clear prepared
        
    fprintf('N = %6d, d = %d | S = %2d, |A(M)| = %6d, R = %1d, Error = %.3e\n', ...
        N, d, S_val, nA, max_len, err_val);
        
    res_errors(k)  = err_val;
    res_S_vals(k)  = S_val;
    res_nA(k)      = nA;
    res_max_len(k) = max_len;
    res_recovery_time(k) = recovery_time;
    res_evaluation_time(k) = evaluation_time;
    fprintf('  Setup: %.3f s | Recovery/shift: %.3f s | Evaluation/shift: %.3f s\n', ...
        res_setup_time(k), recovery_time, evaluation_time);
    clear B A fibers base_lattice_pts
end
total_time = toc(total_clock);
fprintf('Total experiment time (including error evaluation): %.2f seconds\n', total_time);

%% 4. Data Restructuring & Archiving
error_matrix  = reshape(res_errors, num_dims, num_N);
s_matrix      = reshape(res_S_vals, num_dims, num_N);
nA_matrix     = reshape(res_nA, num_dims, num_N);
max_len_matrix= reshape(res_max_len, num_dims, num_N);
frequency_time_matrix = reshape(res_frequency_time, num_dims, num_N);
setup_time_matrix = reshape(res_setup_time, num_dims, num_N);
recovery_time_matrix = reshape(res_recovery_time, num_dims, num_N);
evaluation_time_matrix = reshape(res_evaluation_time, num_dims, num_N);

if error_metric == 2
    metric_str = 'L2';
else
    metric_str = 'Linf';
end
alpha_str = sprintf('alpha%d', floor(alpha));
func_label = sprintf('$f_%d$', floor(alpha) - 1); % Maps alpha=2.5 to f_1, 1.5 to f_2

%% Save result
if ~exist('results', 'dir'), mkdir('results'); end
filename = sprintf('results/korobov_approx_%s_%s_eval%d.mat', alpha_str, metric_str, num_eval_pts);
save(filename, 'dim_list', 'alpha', 'N_list', 'num_eval_pts', 'error_matrix', 's_matrix', 'nA_matrix', 'max_len_matrix', ...
    'frequency_time_matrix', 'setup_time_matrix', 'recovery_time_matrix', 'evaluation_time_matrix');

%% Load result
% if error_metric == 2
%     metric_str = 'L2';
% else
%     metric_str = 'Linf';
% end
% alpha_str = sprintf('alpha%d', floor(alpha));
% func_label = sprintf('$f_%d$', floor(alpha) - 1);
% filename = sprintf('results/korobov_approx_%s_%s_eval%d.mat', alpha_str, metric_str, num_eval_pts);
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
save_figure_eps(gcf, sprintf('figs/korobov_approx_%s_%s_eval%d_N.eps', alpha_str, metric_str, num_eval_pts));

% --- Plot 2: Convergence wrt Total Cost (N * S) ---
figure('Position', [150, 150, 600, 500]);
plot_total_error(alpha, num_target_dims, filtered_dim_list, filtered_s, N_list, filtered_errors, [], error_metric, 9);
title(sprintf('Convergence for %s ($\\alpha = %.1f$)', func_label, alpha), 'Interpreter', 'latex', 'FontSize', 16);
save_figure_eps(gcf, sprintf('figs/korobov_approx_%s_%s_eval%d_tot.eps', alpha_str, metric_str, num_eval_pts));

%% --- Plot 3: Quadratic Scaling of Shifts S wrt log2(N) ---
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

save_figure_eps(gcf, sprintf('figs/korobov_approx_%s_%s_eval%d_S_N.eps', alpha_str, metric_str, num_eval_pts));

%% 6. Running times
% Recovery is the mean time per global shift; the sum excludes Evaluation.
target_dims = [2,4];
figure('Position', [250, 250, 1050, 450]);
for j = 1:length(target_dims)
    d_idx = find(dim_list == target_dims(j));
    setup_times = setup_time_matrix(d_idx, :);
    recovery_times = recovery_time_matrix(d_idx, :);

    subplot(1, 2, j);
    loglog(N_list, setup_times, 'bo-', ...
        N_list, recovery_times, 'rs-', ...
        N_list, setup_times + recovery_times, 'k^-', ...
        'LineWidth', 1.5, 'MarkerSize', 6);
    axis padded;
    grid on; box on;
    set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', 12);
    xlabel('$N$', 'Interpreter', 'latex', 'FontSize', 14);
    ylabel('Time (s)', 'Interpreter', 'latex', 'FontSize', 14);
    title(sprintf('$d=%d$', target_dims(j)), 'Interpreter', 'latex', 'FontSize', 16);
    legend({'Setup', 'Recovery', 'Setup + recovery'}, ...
        'Interpreter', 'latex', 'Location', 'northwest', 'FontSize', 11);
end
save_figure_eps(gcf, sprintf('figs/korobov_approx_%s_%s_eval%d_runtime.eps', alpha_str, metric_str, num_eval_pts));

% =========================================================================
% Local Helper Functions
% =========================================================================
function save_figure_eps(fig_handle, file_path)
    set(fig_handle, 'Color', 'w');
    set(fig_handle, 'PaperPositionMode', 'auto');
    print(fig_handle, file_path, '-depsc', '-vector');
end
