% RUN_COSINE_PDE_APPROXIMATION Evaluates the spectral PDE solver.
%
% Purpose:
%   Reproduces the numerical approximation results for the
%   Poisson problem with homogeneous Neumann boundary conditions.
%   It evaluates the approximation error in the non-periodic half-period
%   cosine space C_{d,\alpha,\gamma} using the tent-transformed
%   deterministic multiple-shift lattice algorithm.

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
N_list = zeros(size(N_list_pool));
for i = 1:num_N
    N0 = N_list_pool(i);
    N_list(i) = nearprime(N0);
end

% Algorithm & PDE Parameters
alpha = 1.5;                % Smoothness parameter equivalent for the cosine space
error_metric = 2;           % Metric selector: 2 for L_2, inf for L_\infty
threshold_t = 0.95;         % Tolerance controlling the condition number bound
num_mc_shifts = 10;         % Number of random shifts for robust MC error evaluation
num_eval_pts = 2^18;        % Number of out-of-sample Sobol points in Section 8

tent_map = @(z) 1 - abs(2 * z - 1);

%% 2. Task Preparation
num_tasks = num_dims * num_N;
[task_n_idx_grid, task_d_idx_grid] = meshgrid(1:num_N, 1:num_dims);
task_n_indices = task_n_idx_grid(:);
task_d_indices = task_d_idx_grid(:);

% Pre-allocate results in task order
res_errors_f = zeros(num_tasks, 1);
res_errors_u = zeros(num_tasks, 1);
res_S_vals   = zeros(num_tasks, 1);
res_nA       = zeros(num_tasks, 1);
res_max_len  = zeros(num_tasks, 1);
res_frequency_time = zeros(num_tasks, 1);
res_setup_time = zeros(num_tasks, 1);
res_recovery_time = zeros(num_tasks, 1);
res_spectral_time = zeros(num_tasks, 1);
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

    % Exact fields, their L2 norms, and the prescribed solution mean
    target_u = @(x) prod(evaluate_1d_components(x, gamma), 2);
    target_f = @(x) evaluate_laplacian(x, gamma);
    [norm_u, norm_f] = compute_analytical_pde_norms(gamma);
    u_0_mean = prod((1 + 20 * gamma) ./ 630);

    % Generate randomized out-of-sample evaluation points via Sobol sequence
    sobol_seq = scramble(sobolset(d), 'MatousekAffineOwen');
    eval_pts = net(sobol_seq, num_eval_pts);

    % Generate random global shifts for error estimation
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
    clear fibers A

    % The cosine grouping and Laplacian eigenvalues are shared by all shifts.
    [prepared.cosine_freqs, ~, prepared.cosine_groups] = unique(abs(prepared.frequencies), 'rows');
    prepared.basis_factor = 1.0 ./ (sqrt(2) .^ sum(prepared.cosine_freqs > 0, 2));
    sum_k_sq = sum(prepared.cosine_freqs.^2, 2);
    prepared.nonzero_modes = sum_k_sq > 0;
    prepared.laplacian_eigenvalues = pi^2 * sum_k_sq(prepared.nonzero_modes);
    res_setup_time(k) = toc(setup_clock);

    % Sample and recover using the same precomputation for every global shift.
    [err_f, err_u, recovery_time, spectral_time, evaluation_time] = evaluate_pde_approximation_error( ...
        g, Y, prepared, num_mc_shifts, mc_shifts, ...
        target_u, target_f, tent_map, u_0_mean, norm_u, norm_f, eval_pts, error_metric);
    clear prepared

    res_errors_f(k) = err_f;
    res_errors_u(k) = err_u;
    res_S_vals(k)   = S_val;
    res_nA(k)       = nA;
    res_max_len(k)  = max_len;
    res_recovery_time(k) = recovery_time;
    res_spectral_time(k) = spectral_time;
    res_evaluation_time(k) = evaluation_time;

    fprintf('N = %6d, d = %d | S = %2d, |A(M)| = %6d, R = %1d | Err(f) = %.3e, Err(u) = %.3e\n', ...
        N, d, S_val, nA, max_len, err_f, err_u);
    fprintf(['  Frequency: %.3f s | Setup: %.3f s | Recovery/shift: %.3f s' ...
        ' | Spectral/shift: %.3f s | Evaluation/shift: %.3f s\n'], ...
        res_frequency_time(k), res_setup_time(k), recovery_time, spectral_time, evaluation_time);
end
total_time = toc(total_clock);
fprintf('Total experiment time (including error evaluation): %.2f seconds\n', total_time);

%% 4. Data Restructuring & Archiving
error_matrix_f = reshape(res_errors_f, num_dims, num_N);
error_matrix_u = reshape(res_errors_u, num_dims, num_N);
s_matrix       = reshape(res_S_vals, num_dims, num_N);
nA_matrix      = reshape(res_nA, num_dims, num_N);
max_len_matrix = reshape(res_max_len, num_dims, num_N);
frequency_time_matrix = reshape(res_frequency_time, num_dims, num_N);
setup_time_matrix = reshape(res_setup_time, num_dims, num_N);
recovery_time_matrix = reshape(res_recovery_time, num_dims, num_N);
spectral_time_matrix = reshape(res_spectral_time, num_dims, num_N);
evaluation_time_matrix = reshape(res_evaluation_time, num_dims, num_N);

if error_metric == 2
    metric_str = 'L2';
else
    metric_str = 'Linf';
end
alpha_str = ['alpha', strrep(sprintf('%g', alpha), '.', '_')];

%% Save result
if ~exist('results', 'dir'), mkdir('results'); end
filename = sprintf('results/cosine_pde_u_%s_%s_eval%d.mat', alpha_str, metric_str, num_eval_pts);
save(filename, 'dim_list', 'alpha', 'N_list', 'num_eval_pts', 'num_mc_shifts', 'error_metric', ...
    'error_matrix_f', 'error_matrix_u', 's_matrix', ...
    'nA_matrix', 'max_len_matrix', 'frequency_time_matrix', 'setup_time_matrix', ...
    'recovery_time_matrix', 'spectral_time_matrix', 'evaluation_time_matrix');
fprintf('Results successfully saved to %s\n', filename);

%% 5. Visualization
if ~exist('figs', 'dir'), mkdir('figs'); end

% Extract data specifically for dimensions d=2 and d=4 as shown in the paper
target_dims = [2, 4];
target_indices = find(ismember(dim_list, target_dims));
num_target_dims = length(target_dims);
filtered_dim_list = dim_list(target_indices);

% Filter error matrices and shifts for target dimensions
filtered_errors_f = error_matrix_f(target_indices, :);
filtered_errors_u = error_matrix_u(target_indices, :);
filtered_s = s_matrix(target_indices, :);
ref_idx = min(8, num_N);

% --- Plot 1: Convergence wrt Base Lattice Size N ---
% Convergence for f(x)
figure('Position', [100, 100, 600, 500]);
plot_lattice_error(alpha, num_target_dims, filtered_dim_list, N_list, filtered_errors_f, [], error_metric, ref_idx);
title(sprintf('Convergence for $f(x)$ ($\\alpha = %.1f$)', alpha), 'Interpreter', 'latex', 'FontSize', 16);
save_figure_eps(gcf, sprintf('figs/cosine_approx_f_%s_%s_N.eps', alpha_str, metric_str));

% Convergence for u(x)
figure('Position', [150, 150, 600, 500]);
plot_lattice_error(alpha, num_target_dims, filtered_dim_list, N_list, filtered_errors_u, [], error_metric, ref_idx);
title(sprintf('Convergence for $u(x)$ ($\\alpha = %.1f$)', alpha), 'Interpreter', 'latex', 'FontSize', 16);
save_figure_eps(gcf, sprintf('figs/cosine_approx_u_%s_%s_N.eps', alpha_str, metric_str));

% --- Plot 2: Convergence wrt Total Cost (N * S) ---
% Convergence for f(x) vs N*S
figure('Position', [200, 200, 600, 500]);
plot_total_error(alpha, num_target_dims, filtered_dim_list, filtered_s, N_list, filtered_errors_f, [], error_metric, ref_idx);
title(sprintf('Convergence for $f(x)$ ($\\alpha = %.1f$)', alpha), 'Interpreter', 'latex', 'FontSize', 16);
save_figure_eps(gcf, sprintf('figs/cosine_approx_f_%s_%s_tot.eps', alpha_str, metric_str));

% Convergence for u(x) vs N*S
figure('Position', [250, 250, 600, 500]);
plot_total_error(alpha, num_target_dims, filtered_dim_list, filtered_s, N_list, filtered_errors_u, [], error_metric, ref_idx);
title(sprintf('Convergence for $u(x)$ ($\\alpha = %.1f$)', alpha), 'Interpreter', 'latex', 'FontSize', 16);
save_figure_eps(gcf, sprintf('figs/cosine_approx_u_%s_%s_tot.eps', alpha_str, metric_str));

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

figure('Position', [300, 300, 650, 450]);
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

% Title adjusted to match the Korobov style
title(sprintf('Shift Scaling for Cosine PDE ($\\alpha = %.1f$)', alpha), 'Interpreter', 'latex', 'FontSize', 16);

fprintf('\n--- Quadratic Fit Coefficients (S = a*x^2 + b*x + c) ---\n');
fprintf('d=2: a = %7.4f, b = %7.4f, c = %7.4f\n', p_d2(1), p_d2(2), p_d2(3));
fprintf('d=4: a = %7.4f, b = %7.4f, c = %7.4f\n', p_d4(1), p_d4(2), p_d4(3));

save_figure_eps(gcf, sprintf('figs/cosine_approx_S_N_%s.eps', alpha_str));

%% 6. Running times
% Recovery and the spectral solve are averaged per global shift.
% The total includes Setup and one solve, excluding test-point evaluation.
figure('Position', [250, 250, 1050, 450]);
for j = 1:num_target_dims
    d_idx = target_indices(j);
    setup_times = setup_time_matrix(d_idx, :);
    solve_times = recovery_time_matrix(d_idx, :) + spectral_time_matrix(d_idx, :);

    subplot(1, 2, j);
    loglog(N_list, setup_times, 'bo-', ...
        N_list, solve_times, 'rs-', ...
        N_list, setup_times + solve_times, 'k^-', ...
        'LineWidth', 1.5, 'MarkerSize', 6);
    axis padded;
    grid on; box on;
    set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', 12);
    xlabel('$N$', 'Interpreter', 'latex', 'FontSize', 14);
    ylabel('Time (s)', 'Interpreter', 'latex', 'FontSize', 14);
    title(sprintf('$d=%d$', dim_list(d_idx)), 'Interpreter', 'latex', 'FontSize', 16);
    legend({'Setup', 'Recovery + spectral solve', 'Setup + recovery + spectral solve'}, ...
        'Interpreter', 'latex', 'Location', 'northwest', 'FontSize', 11);
end
save_figure_eps(gcf, sprintf('figs/cosine_approx_%s_%s_eval%d_runtime.eps', alpha_str, metric_str, num_eval_pts));

% =========================================================================
% Mathematical PDE Helper Functions
% =========================================================================

function u_components = evaluate_1d_components(x, gamma)
% EVALUATE_1D_COMPONENTS Computes the 1D PDE solution components.
% A_j(x) = C * (1 - \gamma_j) + \gamma_j * x^2(1-x)^2

C = 1 / 630;
term_poly = (x.^2) .* ((1 - x).^2);
u_components = C + gamma .* (term_poly - C);
end

function f_val = evaluate_laplacian(x, gamma)
% EVALUATE_LAPLACIAN Computes the non-homogeneous source function \nabla^2 u.
% Utilizes strict vectorized indexing to safely compute the division
% u(x) / A_j(x_j) preventing division by zero when \gamma_1 = 1.

[num_pts, dim_d] = size(x);
u_components = evaluate_1d_components(x, gamma);
u_val = prod(u_components, 2);

% Second derivative: A''_j(x) = \gamma_j * (12x^2 - 12x + 2)
ddA_mat = gamma .* (12 * (x.^2) - 12 * x + 2);

R_mat = zeros(num_pts, dim_d);
is_zero = abs(u_components) < eps;
zero_counts = sum(is_zero, 2);

% Vectorized safe division for points without any zero components
safe_idx = (zero_counts == 0);
if any(safe_idx)
    R_mat(safe_idx, :) = u_val(safe_idx) ./ u_components(safe_idx, :);
end

% Exact algebraic recovery for points with exactly one zero component
one_zero_idx = find(zero_counts == 1);
for i = 1:length(one_zero_idx)
    r = one_zero_idx(i);
    z_col = find(is_zero(r, :));
    idx_others = [1:z_col-1, z_col+1:dim_d];
    R_mat(r, z_col) = prod(u_components(r, idx_others));
end

% Note: If a point has >= 2 zeros, all terms in the Laplacian product
% contain at least one zero. R_mat correctly remains 0.

f_val = sum(ddA_mat .* R_mat, 2);
end

function [norm_u, norm_f] = compute_analytical_pde_norms(gamma)
% COMPUTE_ANALYTICAL_PDE_NORMS Analytically calculates L_2 norms.

C = 1 / 630;
I_P = 1 / 30;
I_P2 = 1 / 630;

% M0: ||A_j||^2
term1 = (C * (1 - gamma)).^2;
term2 = 2 * C * gamma .* (1 - gamma) * I_P;
term3 = (gamma.^2) * I_P2;
M0 = term1 + term2 + term3;

norm_u = sqrt(prod(M0));

% M1: ||A''_j||^2
M1 = (gamma.^2) * (4 / 5);

% M2: <A''_j, A_j>
M2 = (gamma.^2) * (-2 / 105);

ratios_1 = M1 ./ M0;
ratios_2 = M2 ./ M0;

sum_r1 = sum(ratios_1);
sum_cross = sum(ratios_2)^2 - sum(ratios_2.^2);

norm_f = norm_u * sqrt(sum_r1 + sum_cross);
end

function save_figure_eps(fig_handle, file_path)
set(fig_handle, 'Color', 'w');
set(fig_handle, 'PaperPositionMode', 'auto');
print(fig_handle, file_path, '-depsc', '-vector');
end
