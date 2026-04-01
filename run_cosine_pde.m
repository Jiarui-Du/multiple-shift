% RUN_COSINE_PDE_APPROXIMATION Evaluates the spectral PDE solver.
%
% Purpose:
%   Reproduces the numerical approximation results (Section 8.2) for the
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
rng(2026); % Fix random seed for strict reproducibility

%% 1. Experimental Parameters Setup
dim_list = [2,4];
num_dims = length(dim_list);

N_list_pool = 2.^(8:18);
num_N = length(N_list_pool);
for i = 1:num_N
    N0 = N_list_pool(i);
    N_list(i) = nearprime(N0);
end

% Algorithm & PDE Parameters
alpha = 1.5;                % Smoothness parameter equivalent for the cosine space
error_metric = 2;         % Metric selector: 2 for L_2, inf for L_\infty
threshold_t = 0.95;         % Tolerance controlling the condition number bound
num_mc_shifts = 10;         % Number of random shifts for robust MC error evaluation
num_eval_pts = 500000;      % Number of out-of-sample points for evaluation

%% 2. Parallel Task Preparation (Flattening for parfor)
num_tasks = num_dims * num_N;
[task_n_idx_grid, task_d_idx_grid] = meshgrid(1:num_N, 1:num_dims);
task_n_indices = task_n_idx_grid(:);
task_d_indices = task_d_idx_grid(:);

res_errors_f = zeros(num_tasks, 1);
res_errors_u = zeros(num_tasks, 1);
res_S_vals   = zeros(num_tasks, 1);
res_nA       = zeros(num_tasks, 1);
res_max_len  = zeros(num_tasks, 1);

%% 3. Main Parallel Computation Loop
tic;
for k = 1:num_tasks
    n_idx = task_n_indices(k);
    d_idx = task_d_indices(k);

    N = N_list(n_idx);
    d = dim_list(d_idx);

    % Weights \gamma_j specifically designed for the PDE experiment
    gamma = 2.^((1 - (1:d)) / 10);

    % Compute exact analytical L2 norms for the relative error calculation
    [norm_u, norm_f] = compute_analytical_pde_norms(gamma);

    % Mean value \hat{u}(0) required to fix the Neumann boundary solution
    u_0_mean = prod((1 + 20 * gamma) ./ 630);

    % Define the tent transformation operator \psi(z)
    tent_map = @(z) 1 - abs(2 * z - 1);

    % Define the exact target functions in the spatial domain
    target_u = @(x) prod(evaluate_1d_components(x, gamma), 2);
    target_f = @(x) evaluate_laplacian(x, gamma);

    % Generate randomized evaluation points via Sobol sequence
    sobol_seq = scramble(sobolset(d), 'MatousekAffineOwen');
    eval_pts = net(sobol_seq, num_eval_pts);
    mc_shifts = rand(d, num_mc_shifts);

    M = find_M_optimal(N, d, alpha, gamma);

    [err_u, err_f, S_val, nA, max_len] = evaluate_pde_approximation_error( ...
        N, M, d, alpha, gamma, threshold_t, num_mc_shifts, mc_shifts, ...
        target_u, target_f, tent_map, u_0_mean, norm_u, norm_f, eval_pts, error_metric);

    fprintf('N = %6d, d = %d | S = %3d, |A(M)| = %6d, R = %3d | Err(f) = %.3e, Err(u) = %.3e\n', ...
        N, d, S_val, nA, max_len, err_f, err_u);

    res_errors_f(k) = err_f;
    res_errors_u(k) = err_u;
    res_S_vals(k)   = S_val;
    res_nA(k)       = nA;
    res_max_len(k)  = max_len;
end
total_time = toc;
fprintf('Total parfor loop time: %.2f seconds\n', total_time);

%% 4. Data Restructuring & Archiving
error_matrix_f = reshape(res_errors_f, num_dims, num_N);
error_matrix_u = reshape(res_errors_u, num_dims, num_N);
s_matrix       = reshape(res_S_vals, num_dims, num_N);
nA_matrix      = reshape(res_nA, num_dims, num_N);
max_len_matrix = reshape(res_max_len, num_dims, num_N);

%% Save result
if ~exist('results', 'dir'), mkdir('results'); end
filename = sprintf('results/cosine_pde_u_%s_%s.mat', alpha_str, metric_str);
save(filename, 'dim_list', 'alpha', 'N_list', ...
    'error_matrix_f', 'error_matrix_u', 's_matrix', ...
    'nA_matrix', 'max_len_matrix');
fprintf('Results successfully saved to %s\n', filename);

%% Load result
% if error_metric == 2
%     metric_str = 'L2';
% else
%     metric_str = 'Linf';
% end
% alpha_str = sprintf('alpha%d', floor(alpha));
% func_label = sprintf('$f_%d$', floor(alpha) - 1);
% filename = sprintf('results/cosine_pde_u_%s_%s.mat', alpha_str, metric_str);
% load(filename)

%% 5. Visualization (Publishing Standard)
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

% =========================================================================
% --- Plot Set 1: Convergence wrt Base Lattice Size N ---
% =========================================================================

% Convergence for f(x)
figure('Position', [100, 100, 600, 500]);
plot_lattice_error(alpha, num_target_dims, filtered_dim_list, N_list, filtered_errors_f, [], error_metric, 8);
title(sprintf('Convergence for $f(x)$ ($\\alpha = %.1f$)', alpha), 'Interpreter', 'latex', 'FontSize', 16);
save_figure_eps(gcf, sprintf('figs/cosine_approx_f_%s_%s_N.eps', alpha_str, metric_str));

% Convergence for u(x)
figure('Position', [150, 150, 600, 500]);
plot_lattice_error(alpha, num_target_dims, filtered_dim_list, N_list, filtered_errors_u, [], error_metric, 8);
title(sprintf('Convergence for $u(x)$ ($\\alpha = %.1f$)', alpha), 'Interpreter', 'latex', 'FontSize', 16);
save_figure_eps(gcf, sprintf('figs/cosine_approx_u_%s_%s_N.eps', alpha_str, metric_str));

% =========================================================================
% --- Plot Set 2: Convergence wrt Total Cost (N * S) ---
% =========================================================================

% Convergence for f(x) vs N*S
figure('Position', [200, 200, 600, 500]);
plot_total_error(alpha, num_target_dims, filtered_dim_list, filtered_s, N_list, filtered_errors_f, [], error_metric, 8);
title(sprintf('Convergence for $f(x)$ ($\\alpha = %.1f$)', alpha), 'Interpreter', 'latex', 'FontSize', 16);
save_figure_eps(gcf, sprintf('figs/cosine_approx_f_%s_%s_tot.eps', alpha_str, metric_str));

% Convergence for u(x) vs N*S
figure('Position', [250, 250, 600, 500]);
plot_total_error(alpha, num_target_dims, filtered_dim_list, filtered_s, N_list, filtered_errors_u, [], error_metric, 8);
title(sprintf('Convergence for $u(x)$ ($\\alpha = %.1f$)', alpha), 'Interpreter', 'latex', 'FontSize', 16);
save_figure_eps(gcf, sprintf('figs/cosine_approx_u_%s_%s_tot.eps', alpha_str, metric_str));

% =========================================================================
% --- Plot Set 3: Quadratic Scaling of Shifts S wrt log2(N) ---
% =========================================================================
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